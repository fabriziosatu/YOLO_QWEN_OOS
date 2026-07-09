"""
qwen_inference_module_v2.py  —  v2 (v6.4)
-------------------------------------------
Initialization of the Qwen2.5-VL model and management of the inference process.

COORDINATE TRACKING CHARACTERISTICS:
  - Resolved spatial misalignment by computing coordinate conversions relative to the original image dimensions, 
    ensuring scaling tracking remains completely independent of processing-level resizing actions.
"""

from __future__ import annotations
import torch
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a visual inspection assistant for a retail inventory monitoring system. "
    "Your task is to determine whether a specific shelf area is empty (no products present). "
    "You will be shown a shelf image with a green bounding box highlighting the area of interest. "
    "Be decisive: if the area looks empty or you are uncertain, answer YES. "
    "Answer only with 'yes' or 'no'. Do not add any other words."
)

SYSTEM_PROMPT_CROP_ONLY = (
    "You are a visual inspection assistant for a retail inventory monitoring system. "
    "Your task is to determine whether a specific shelf area is empty (no products present). "
    "You will be shown a close-up crop of the shelf area to inspect. "
    "Be decisive: if the area looks empty or you are uncertain, answer YES. "
    "Answer only with 'yes' or 'no'. Do not add any other words."
)

# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT TEMPLATES
#  {x1} {y1} {x2} {y2} are replaced with coordinates scaled to [0, 1000]
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATES: dict[str, str] = {

    "no_context": (
        "The image shows a supermarket shelf. "
        "A green bounding box at coordinates <box>({x1},{y1}),({x2},{y2})</box> "
        "highlights the area of interest. "
        "Is the highlighted shelf area empty (no products)? "
        "If you are unsure, answer yes. "
        "Answer yes or no."
    ),

    "context": (
        "The image shows a supermarket shelf. "
        "A green bounding box at coordinates <box>({x1},{y1}),({x2},{y2})</box> "
        "highlights the area of interest.\n\n"
        "Your goal is to detect Out-Of-Stock (OOS) shelf gaps.\n\n"
        "Answer YES if:\n"
        "  - The area is clearly empty (bare shelf, no products visible)\n"
        "  - You are uncertain or cannot clearly determine whether products are present\n\n"
        "Answer NO only if you can clearly and confidently see that the area contains "
        "products or is occupied by shelf structure (uprights, dividers, refrigerator doors).\n\n"
        "When in doubt, answer YES.\n"
        "Answer yes or no."
    ),

    # ── Fallback crop-only templates (used after OOM) ────────────────────────
    "no_context_crop": (
        "This is a close-up crop of a supermarket shelf area. "
        "Is this shelf area empty (no products)? "
        "If you are unsure, answer yes. "
        "Answer yes or no."
    ),

    "context_crop": (
        "This is a close-up crop of a supermarket shelf area. "
        "Your goal is to detect Out-Of-Stock (OOS) shelf gaps.\n\n"
        "Answer YES if:\n"
        "  - The area is clearly empty (bare shelf, no products visible)\n"
        "  - You are uncertain or cannot clearly determine whether products are present\n\n"
        "Answer NO only if you can clearly and confidently see products or shelf "
        "structure filling the area.\n\n"
        "When in doubt, answer YES.\n"
        "Answer yes or no."
    ),
}
# ─────────────────────────────────────────────────────────────────────────────

COLOR_TARGET  = (0, 255, 0)
BB_WIDTH      = 3
CROP_PADDING  = 0.175
MAX_FULL_SIZE = 1280


def _resize_if_needed(img: Image.Image, max_side: int = MAX_FULL_SIZE) -> Image.Image:
    """Rescales the image proportionally if the longest side exceeds max_side constraints."""
    W, H = img.size
    if max(W, H) <= max_side:
        return img
    scale = max_side / max(W, H)
    return img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)


def _draw_target_box(img: Image.Image, box: list[float]) -> Image.Image:
    """
    Resizes the input image to MAX_FULL_SIZE and draws the target green bounding box.
    The box coordinates are provided in absolute pixels relative to the original image size.
    """
    img_r = _resize_if_needed(img)
    W_o, H_o = img.size
    W_r, H_r = img_r.size
    sx, sy = W_r / W_o, H_r / H_o

    out  = img_r.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    x1 = max(0,        int(box[0] * sx))
    y1 = max(0,        int(box[1] * sy))
    x2 = min(W_r - 1,  int(box[2] * sx))
    y2 = min(H_r - 1,  int(box[3] * sy))
    if x2 > x1 and y2 > y1:
        draw.rectangle([x1, y1, x2, y2], outline=COLOR_TARGET, width=BB_WIDTH)
    return out


# Minimum dimensions required by the Qwen processor (patch size = 28px)
CROP_MIN_SIZE = 28

def _extract_crop(img: Image.Image, box: list[float]) -> Image.Image:
    """Extracts the bounding box crop with 17.5% padding, guaranteeing a minimum of 28px per side."""
    W, H = img.size
    bw   = box[2] - box[0]
    bh   = box[3] - box[1]
    cx1  = max(0,     int(box[0] - bw * CROP_PADDING))
    cy1  = max(0,     int(box[1] - bh * CROP_PADDING))
    cx2  = min(W - 1, int(box[2] + bw * CROP_PADDING))
    cy2  = min(H - 1, int(box[3] + bh * CROP_PADDING))
    crop = img.crop((cx1, cy1, cx2, cy2)).convert("RGB")
    cw, ch = crop.size
    if cw < CROP_MIN_SIZE or ch < CROP_MIN_SIZE:
        scale = max(CROP_MIN_SIZE / max(cw, 1), CROP_MIN_SIZE / max(ch, 1))
        crop  = crop.resize(
            (max(int(cw * scale), CROP_MIN_SIZE),
             max(int(ch * scale), CROP_MIN_SIZE)),
            Image.LANCZOS,
        )
    return crop


def _qwen_coords(box: list[float], W: int, H: int) -> dict[str, str]:
    """
    Converts absolute pixel coordinates into a [0, 1000] scale for Qwen2.5-VL's native
    <box> tags. This represents the native grounding format the model was trained on,
    providing higher spatial accuracy compared to generic standard [0,1] coordinates.
    """
    return {
        "x1": str(int((box[0] / W) * 1000)),
        "y1": str(int((box[1] / H) * 1000)),
        "x2": str(int((box[2] / W) * 1000)),
        "y2": str(int((box[3] / H) * 1000)),
    }


def _parse_response(raw: str, box: list[float]) -> str:
    """
    Parses Qwen's textual response string.
    Returns 'yes' or 'no'. Applies a default confirmation bias ('yes') for ambiguous outputs.
    """
    cleaned = raw.strip().lower()

    if cleaned in ("yes", "yes.", "yes!", "sì", "si"):
        return "yes"
    if cleaned in ("no", "no.", "no!"):
        return "no"
    if cleaned.startswith("yes"):
        return "yes"
    if cleaned.startswith("no"):
        return "no"

    print(f"  [AMBIGUOUS] Uninterpretable response string: '{raw}' "
          f"for box {[round(b, 1) for b in box]} -> falling back to 'yes'")
    return "yes"


class QwenInferenceModule:
    """
    Loads the frozen Qwen2.5-VL model and uses it to evaluate shelf bounding boxes.

    For each bounding box, the model receives:
      - A full-size image (max 1280px) overlayed with a green target bounding box
      - A text prompt embedding the normalized [0,1000] scale coordinates of the box

    Response: 'yes' (empty shelf region) or 'no' (non-empty shelf region).
    Ambiguous outputs are caught, logged, and fallback handled as an automated 'yes'.
    If an OOM occurs, it triggers a standalone crop verification workflow without localizing text.
    """

    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = torch_dtype

        print(f"  [Qwen] Loading processor from {model_path}")
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        print(f"  [Qwen] Loading model ({torch_dtype}) ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"  [Qwen] Model ready on {self.device}")

    def _forward_text(
        self,
        images: list[Image.Image],
        prompt_text: str,
        system_prompt: str,
    ) -> str:
        """
        Executes a targeted vision-language forward pass for text generation.
        Returns the raw model output string.
        """
        user_content = [
            {"type": "text",  "text": "Shelf image with bounding box:"},
            {"type": "image"},
            {"type": "text",  "text": prompt_text},
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def score_boxes(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        prompt_name: str = "no_context",
    ) -> list[str]:
        """
        Returns Qwen's textual classification response ('yes'/'no') for each item in boxes.

          'yes' -> Qwen confirms that the targeted bounding box area is empty
          'no'  -> Qwen rejects the YOLO candidate suggestion
          Ambiguous -> handled as 'yes' (confirmation bias)

        Input: Full image containing green BB + coordinate scaling to [0, 1000].
        Failsafes: Switches to single crops automatically without global coordinates if VRAM hits an OOM.
        """
        if not boxes:
            return []

        W, H          = image.size
        template      = PROMPT_TEMPLATES[prompt_name]
        template_crop = PROMPT_TEMPLATES[f"{prompt_name}_crop"]
        responses     = []
        oom_count     = 0

        for box in boxes:
            annotated = _draw_target_box(image, box)

            # Convert absolute pixel coordinates to Qwen's native [0, 1000] scale.
            # Using the original W and H guarantees resolution resize tracking independence.
            coords = {
                "x1": str(int((box[0] / W) * 1000)),
                "y1": str(int((box[1] / H) * 1000)),
                "x2": str(int((box[2] / W) * 1000)),
                "y2": str(int((box[3] / H) * 1000)),
            }
            prompt = template.format(**coords)

            try:
                raw = self._forward_text(
                    images        = [annotated],
                    prompt_text   = prompt,
                    system_prompt = SYSTEM_PROMPT,
                )

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                torch.cuda.empty_cache()
                print(f"  [OOM] Fallback to standalone crop mode for box {[round(b,1) for b in box]} "
                      f"(Total OOM occurrences: {oom_count})")
                crop = _extract_crop(image, box)
                try:
                    raw = self._forward_text(
                        images        = [crop],
                        prompt_text   = template_crop,
                        system_prompt = SYSTEM_PROMPT_CROP_ONLY,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  [OOM] Crop mode also encountered OOM -> forcing fallback response 'yes'")
                    raw = "yes"

            responses.append(_parse_response(raw, box))
            torch.cuda.empty_cache()

        if oom_count > 0:
            print(f"  [OOM] Total OOM episodes caught: {oom_count}")

        return responses
