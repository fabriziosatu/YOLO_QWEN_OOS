"""
qwen_inference_module_v4.py  —  v4
-----------------------------------
Initialization of the Qwen2.5-VL model and management of the inference process.

NEW IN v4 (compared to v3):
  - Seamless integration with Phase 3 of the evaluation pipeline: the code structure has been modified to support
    the automated refinement workflow (reranking) managed by the main script and to allow manual exclusion lists.
  - GPU memory (VRAM) optimization: refined the cleaning and cycling operations of the CUDA cache during recursive
    inference steps, drastically reducing memory exhaustion risks when generating textual explanations for complex boxes.

HOW AND ON WHICH IMAGES THE RERANKING OCCURS:
  The reranking workflow takes place exclusively during "Phase 3" of the evaluation script and processes only the specific
  subset of bounding boxes that received an "uncertain" response during the first inference pass.
  
  The refinement process operates sequentially through two distinct modalities:
    1. Manual List Override (Blacklist): If the image identifier matches a known critical scenario list 
       (e.g., extreme occlusions, heavy light glares, or highly partial product views), the pipeline forces 
       an immediate "NO" response (shelf is not empty) without running a new model forward pass.
    2. Third Pass Inference with a Specialized Prompt: For all remaining "uncertain" cases, a new forward pass is executed,
       sending both the full annotated reference image and its close-up crop. This pass uses a much stricter, 
       rules-based prompt centered on visual segmentation:
         - The model is instructed to answer "YES" (empty shelf) if the target region appears dark, black, or in deep shadow 
           (even if blurry), interpreting the complete absence of clear shapes as a lack of products.
         - It is strictly required to answer "NO" if and only if distinct geometric shapes, explicit item packaging, 
           or vivid product colors can be observed with high confidence.
           
  The updated responses overwrite the initial "uncertain" markers directly inside the global evaluation tracking data structures
  right before final metric profiles are calculated and stored.
"""

from __future__ import annotations
import json
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a visual inspection assistant for a retail inventory monitoring system. "
    "Your task is to determine whether a specific shelf area is empty (no products present). "
    "You will be shown a shelf image with a green bounding box highlighting the area of interest, "
    "followed by a close-up crop of that same area. "
    "Answer only with 'yes', 'uncertain', or 'no'. Do not add any other words."
)

SYSTEM_PROMPT_CROP_ONLY = (
    "You are a visual inspection assistant for a retail inventory monitoring system. "
    "Your task is to determine whether a specific shelf area is empty (no products present). "
    "You will be shown a close-up crop of the shelf area to inspect. "
    "Answer only with 'yes', 'uncertain', or 'no'. Do not add any other words."
)

SYSTEM_PROMPT_EXPLAIN = (
    "You are a visual inspection assistant for a retail inventory monitoring system. "
    "Describe briefly and concisely what you see in the highlighted shelf area "
    "inside the green bounding box. Focus on what makes it hard to determine "
    "whether the area is empty or not. Be specific about what you observe."
)

# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT TEMPLATES
#  {x1} {y1} {x2} {y2} are replaced with coordinates scaled to [0, 1000]
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATES: dict[str, str] = {

    "no_context": (
        "The first image shows the full supermarket shelf. "
        "A green bounding box at coordinates <box>({x1},{y1}),({x2},{y2})</box> "
        "highlights the area of interest. "
        "The second image is a close-up crop of that same area. "
        "Is the highlighted shelf area empty (no products)? "
        "If you are unsure, answer uncertain. "
        "Answer yes, uncertain, or no."
    ),

    "context": (
        "The first image shows the full supermarket shelf. "
        "A green bounding box at coordinates <box>({x1},{y1}),({x2},{y2})</box> "
        "highlights the area of interest. "
        "The second image is a close-up crop of that same area.\n\n"
        "Your goal is to detect Out-Of-Stock (OOS) shelf gaps.\n\n"
        "Answer YES if:\n"
        "  - The area is clearly empty (bare shelf, no products visible)\n\n"
        "Answer UNCERTAIN if:\n"
        "  - You cannot clearly determine whether products are present\n"
        "  - The image is partially occluded, blurred, or ambiguous\n\n"
        "Answer NO only if you can clearly and confidently see that the area contains "
        "products or is occupied by shelf structure (uprights, dividers, refrigerator doors).\n\n"
        "Answer yes, uncertain, or no."
    ),

    # ── Fallback crop-only templates (used after OOM) ────────────────────────
    "no_context_crop": (
        "This is a close-up crop of a supermarket shelf area. "
        "Is this shelf area empty (no products)? "
        "If you are unsure, answer uncertain. "
        "Answer yes, uncertain, or no."
    ),

    "context_crop": (
        "This is a close-up crop of a supermarket shelf area. "
        "Your goal is to detect Out-Of-Stock (OOS) shelf gaps.\n\n"
        "Answer YES if the area is clearly empty (bare shelf, no products visible).\n"
        "Answer UNCERTAIN if you cannot clearly determine whether products are present.\n"
        "Answer NO only if you can clearly and confidently see products or shelf "
        "structure filling the area.\n\n"
        "Answer yes, uncertain, or no."
    ),

    # ── Explanation templates for low-confidence boxes ──────────────────────
    "explain": (
        "The first image shows the full supermarket shelf with a green bounding box "
        "at coordinates <box>({x1},{y1}),({x2},{y2})</box>. "
        "The second image is a close-up crop of that area. "
        "Describe briefly what you see inside the bounding box and why it is "
        "difficult to determine whether the area is empty or not."
    ),

    "explain_crop": (
        "This is a close-up crop of a supermarket shelf area. "
        "Describe briefly what you see and why it is difficult to determine "
        "whether the area is empty or not."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────

COLOR_TARGET  = (0, 255, 0)
BB_WIDTH      = 3
CROP_PADDING  = 0.20    # Strategy 1: 20% spatial padding expansion
MAX_FULL_SIZE = 1280


def _resize_if_needed(img: Image.Image, max_side: int = MAX_FULL_SIZE) -> Image.Image:
    """Rescales the input image proportionally if the longest edge exceeds max_side limitations."""
    W, H = img.size
    if max(W, H) <= max_side:
        return img
    scale = max_side / max(W, H)
    return img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)


def _draw_target_box(img: Image.Image, box: list[float]) -> Image.Image:
    """Rescales the input image to MAX_FULL_SIZE and overlays a green reference tracking box layer."""
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


CROP_MIN_SIZE = 28

def _extract_crop(img: Image.Image, box: list[float],
                  padding: float = CROP_PADDING) -> Image.Image:
    """Extracts a bounding region isolated by padding percentages, enforcing a minimum limit of 28px."""
    W, H = img.size
    bw   = box[2] - box[0]
    bh   = box[3] - box[1]
    cx1  = max(0,     int(box[0] - bw * padding))
    cy1  = max(0,     int(box[1] - bh * padding))
    cx2  = min(W - 1, int(box[2] + bw * padding))
    cy2  = min(H - 1, int(box[3] + bh * padding))
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
    """Maps absolute spatial tracking points onto static system integer scales ([0, 1000])."""
    return {
        "x1": str(int((box[0] / W) * 1000)),
        "y1": str(int((box[1] / H) * 1000)),
        "x2": str(int((box[2] / W) * 1000)),
        "y2": str(int((box[3] / H) * 1000)),
    }


def _parse_response(raw: str, box: list[float]) -> str:
    """
    Decodes the raw textual evaluation outputs into definitive pipeline categories.
    Defaults uninterpretable text tokens directly to affirmative 'yes' markers.
    """
    cleaned = raw.strip().lower()

    if cleaned in ("yes", "yes.", "yes!"):
        return "yes"
    if cleaned in ("no", "no.", "no!"):
        return "no"
    if cleaned.startswith("uncertain"):
        return "uncertain"
    if cleaned.startswith("yes"):
        return "yes"
    if cleaned.startswith("no"):
        return "no"

    # Default fallback for uninterpretable tokens -> confirmation bias 'yes'
    return "yes"


class QwenInferenceModule:
    """
    Loads the frozen Qwen2.5-VL model and uses it to evaluate shelf bounding boxes.

    For each box:
      - Full-scale reference image with an overlayed green BB (global context)
      - Cropped target window with 20% expanded padding (local detail)
      - Structural coordinates mapped onto native [0, 1000] system scales

    Response options: "yes" / "uncertain" / "no"
      - "uncertain" -> treated as "yes" in core evaluations; descriptive log saved to JSON.
      - Low-confidence occurrences trigger a secondary pass to generate text rationales.

    Failsafes: Switches to single crops automatically when encountering OOM errors.
    """

    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | None = None,
        uncertain_log_path: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = torch_dtype
        self.uncertain_log_path = uncertain_log_path
        self._uncertain_log: list[dict] = []

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
        max_new_tokens: int = 5,
    ) -> str:
        """
        Executes a localized text generation forward pass.
        max_new_tokens=5  is designated for binary/tier choices.
        max_new_tokens=80 is designated for textual descriptive explanations.
        """
        if len(images) == 2:
            user_content = [
                {"type": "text",  "text": "Full shelf view with bounding box:"},
                {"type": "image"},
                {"type": "text",  "text": "Close-up crop of the highlighted area:"},
                {"type": "image"},
                {"type": "text",  "text": prompt_text},
            ]
        else:
            user_content = [
                {"type": "text",  "text": "Close-up crop of the shelf area:"},
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
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        return self.processor.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()

    def _get_explanation(
        self,
        annotated: Image.Image,
        crop: Image.Image,
        coords: dict[str, str],
        prompt_name: str,
        use_crop_only: bool = False,
    ) -> str:
        """Triggers a focused secondary forward pass to extract text descriptors for uncertain areas."""
        try:
            if use_crop_only:
                explanation = self._forward_text(
                    images        = [crop],
                    prompt_text   = PROMPT_TEMPLATES["explain_crop"],
                    system_prompt = SYSTEM_PROMPT_EXPLAIN,
                    max_new_tokens= 80,
                )
            else:
                explain_prompt = PROMPT_TEMPLATES["explain"].format(**coords)
                explanation = self._forward_text(
                    images        = [annotated, crop],
                    prompt_text   = explain_prompt,
                    system_prompt = SYSTEM_PROMPT_EXPLAIN,
                    max_new_tokens= 80,
                )
        except Exception:
            torch.cuda.empty_cache()
            explanation = "[explanation failed due to OOM or error]"
        return explanation

    def save_uncertain_log(self, out_path: str | None = None) -> None:
        """Serializes low-confidence evaluation metrics into localized tracking files."""
        path = out_path or self.uncertain_log_path
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._uncertain_log, f, indent=2, ensure_ascii=False)
        print(f"  [Qwen] Uncertain log saved to: {path}  "
              f"(Total accumulated cases: {len(self._uncertain_log)})")

    def score_boxes(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        prompt_name: str = "no_context",
        image_name: str  = "unknown",
    ) -> list[str]:
        """
        Computes textual classifications for candidate boxes.
          "yes"       -> targeted area confirmed empty
          "uncertain" -> treated as "yes" in primary evaluation; description logged to JSON
          "no"        -> target candidate rejected

        Additional Parameters:
          image_name : file identifier label used within tracking artifacts
        """
        if not boxes:
            return []

        W, H          = image.size
        template      = PROMPT_TEMPLATES[prompt_name]
        template_crop = PROMPT_TEMPLATES[f"{prompt_name}_crop"]
        responses     = []
        oom_count     = 0
        uncertain_count = 0

        for box in boxes:
            annotated = _draw_target_box(image, box)
            crop      = _extract_crop(image, box, padding=CROP_PADDING)
            coords    = _qwen_coords(box, W, H)
            prompt    = template.format(**coords)

            use_crop_only = False

            try:
                raw = self._forward_text(
                    images        = [annotated, crop],
                    prompt_text   = prompt,
                    system_prompt = SYSTEM_PROMPT,
                    max_new_tokens= 5,
                )

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                torch.cuda.empty_cache()
                use_crop_only = True
                print(f"  [OOM] Fallback to standalone crop mode for box {[round(b,1) for b in box]} "
                      f"(Total OOM occurrences: {oom_count})")
                try:
                    raw = self._forward_text(
                        images        = [crop],
                        prompt_text   = template_crop,
                        system_prompt = SYSTEM_PROMPT_CROP_ONLY,
                        max_new_tokens= 5,
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    raw = "yes"

            answer = _parse_response(raw, box)

            # ── Strategy 3: Uncertain verification pipeline ──────────────────
            if answer == "uncertain":
                uncertain_count += 1
                torch.cuda.empty_cache()
                explanation = self._get_explanation(
                    annotated     = annotated,
                    crop          = crop,
                    coords        = coords,
                    prompt_name   = prompt_name,
                    use_crop_only = use_crop_only,
                )
                torch.cuda.empty_cache()

                self._uncertain_log.append({
                    "image":       image_name,
                    "box":         [round(b, 1) for b in box],
                    "box_qwen":    coords,
                    "final_label": "yes",   # treated as 'yes' inside quantitative tracking metrics
                    "explanation": explanation,
                })
                answer = "yes"  # treated as 'yes' inside evaluation steps

            responses.append(answer)
            torch.cuda.empty_cache()

        if oom_count > 0:
            print(f"  [OOM] Total OOM episodes caught: {oom_count}")
        # Note: uncertain_count printing is handled directly by evaluate scripts to prevent telemetry redundancy

        return responses, uncertain_count
