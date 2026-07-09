"""
qwen_inference_module.py  —  v1 (v5.2)
---------------------------------------
Initialization of the Qwen2.5-VL model and management of the inference process.

NEW IN THIS VERSION:
  - Automatic resizing of full images (maximum limit of 1280 pixels on the longest side)
    to prevent GPU memory exhaustion (Out-Of-Memory errors) when processing high-resolution images (Ultra HD or 4K).
  - Proportional adjustment of bounding box coordinates to ensure that the green rectangle drawing is always precise.
  - Safety mechanism for full memory (OOM) scenarios: if the analysis of the full image fails due to low memory,
    the system automatically falls back to analyzing only the cropped portion of the shelf area.
  - Probability computation: extraction and comparison of the numerical logit scores for 'yes' and 'no' tokens
    to accurately determine the probability of the shelf being empty.
"""

from __future__ import annotations
import torch
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT  —  v2: explicit visual criteria
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert visual inspector for a retail out-of-stock detection system. "
    "You analyze surveillance camera images from supermarket aisles to determine "
    "whether a specific shelf section is EMPTY of products.\n\n"
    "A TRUE empty shelf shows: bare horizontal shelf surface (metal, wood, or plastic), "
    "visible shelf backing or pegboard, and clear depth/perspective into the empty space. "
    "There are NO products, packages, boxes, bottles, or any merchandise visible.\n\n"
    "You will see two images of the SAME area: (1) the full shelf photo with a green "
    "box marking the region to inspect, and (2) a zoomed-in crop of that exact region. "
    "Use the crop for detail and the full image for context (shelf depth, lighting, "
    "surrounding products) to make your decision.\n"
    "Respond with exactly one word: 'yes' (the shelf is empty) or 'no' (it is not)."
)

SYSTEM_PROMPT_CROP_ONLY = (
    "You are an expert visual inspector for a retail out-of-stock detection system. "
    "You analyze a cropped photo of a supermarket shelf section to determine whether "
    "it is EMPTY of products.\n\n"
    "A TRUE empty shelf shows: bare horizontal shelf surface, visible shelf backing "
    "or pegboard, and no products, packages, boxes, bottles, or merchandise of any kind.\n"
    "Respond with exactly one word: 'yes' (the shelf is empty) or 'no' (it is not)."
)

# ─────────────────────────────────────────────────────────────────────────────
#  PROMPTS  —  v2: positive definition + negative checklist
# ─────────────────────────────────────────────────────────────────────────────
PROMPTS: dict[str, str] = {

    "no_context": (
        "The first image shows the full shelf view with a green bounding box "
        "highlighting the area of interest. "
        "The second image is a close-up crop of that same area. "
        "Is the highlighted shelf area empty (no products)? "
        "Answer yes or no."
    ),

    "context": (
        "The first image shows the full shelf view with a green bounding box "
        "marking the area of interest. The second image is a zoomed-in crop of "
        "that exact region.\n\n"
        "STEP 1 — Look at the crop. Can you see a flat, bare shelf surface with "
        "visible depth (the back wall, pegboard, or shadow of the empty space)? "
        "If yes, this supports EMPTY.\n\n"
        "STEP 2 — Check the crop does NOT match any of these false-positive cases:\n"
        "  - Shelf STRUCTURE only: a metallic uprights, edge profile, or divider seen "
        "edge-on (this is shelf hardware, not empty space)\n"
        "  - REFRIGERATOR/FREEZER glass door or surface, possibly with reflections\n"
        "  - MOTION BLUR: the crop is blurry/smeared from camera movement, "
        "details cannot be confirmed\n"
        "  - DISTANT/DARK shelf that LOOKS empty due to poor lighting, but the full "
        "image shows products are likely still there (check the surrounding rows)\n"
        "  - PRODUCTS present but small, dark-colored, or partially out of frame\n\n"
        "Answer YES only if step 1 is satisfied AND none of the step 2 cases apply.\n"
        "Answer NO if the crop matches any step 2 case, or if you cannot clearly "
        "confirm bare shelf surface.\n"
        "Answer with exactly one word: yes or no."
    ),

    # Fallback prompt used when only the crop is passed (after OOM)
    "no_context_crop": (
        "This is a close-up crop of a supermarket shelf area. "
        "Is this shelf area empty (no products)? "
        "Answer yes or no."
    ),

    "context_crop": (
        "This is a close-up crop of a supermarket shelf area.\n\n"
        "STEP 1 — Can you see a flat, bare shelf surface with visible depth "
        "(back wall, pegboard, or shadow)? If yes, this supports EMPTY.\n\n"
        "STEP 2 — Check the crop does NOT match any of these false-positive cases:\n"
        "  - Shelf STRUCTURE only: metallic uprights, edge profile, or divider\n"
        "  - REFRIGERATOR/FREEZER glass door or surface\n"
        "  - MOTION BLUR: the crop is blurry/smeared\n"
        "  - DARK shelf that looks empty due to poor lighting only\n"
        "  - PRODUCTS present but small, dark-colored, or partially visible\n\n"
        "Answer YES only if step 1 is satisfied AND none of the step 2 cases apply.\n"
        "Answer NO otherwise.\n"
        "Answer with exactly one word: yes or no."
    ),
}
# ─────────────────────────────────────────────────────────────────────────────

COLOR_TARGET  = (0, 255, 0)
BB_WIDTH      = 3
CROP_PADDING  = 0.175   # 17.5%
MAX_FULL_SIZE = 1280    # Maximum full image side length before feeding it to Qwen
                        # Full HD (1920x1080) -> ~1280x720  (~900 visual tokens)
                        # 4K    (3840x2160)   -> ~1280x720  (prevents OOM errors)


def _resize_if_needed(img: Image.Image, max_side: int = MAX_FULL_SIZE) -> Image.Image:
    """
    Rescales the image proportionally if the longest side exceeds max_side.
    """
    W, H = img.size
    if max(W, H) <= max_side:
        return img
    scale = max_side / max(W, H)
    return img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)


def _draw_target_box(img: Image.Image, box: list[float]) -> Image.Image:
    """
    Resizes the image to MAX_FULL_SIZE and draws the green bounding box
    proportionally rescaled.
    """
    img_r = _resize_if_needed(img)
    W_o, H_o = img.size
    W_r, H_r = img_r.size
    sx, sy = W_r / W_o, H_r / H_o

    out  = img_r.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    x1 = max(0,       int(box[0] * sx))
    y1 = max(0,       int(box[1] * sy))
    x2 = min(W_r - 1, int(box[2] * sx))
    y2 = min(H_r - 1, int(box[3] * sy))
    if x2 > x1 and y2 > y1:
        draw.rectangle([x1, y1, x2, y2], outline=COLOR_TARGET, width=BB_WIDTH)
    return out


# Minimum dimension required by the Qwen processor (patch size = 28px)
CROP_MIN_SIZE = 28

def _extract_crop(img: Image.Image, box: list[float]) -> Image.Image:
    """
    Extracts the crop of the bounding box with a 17.5% padding.
    Guarantees a minimum dimension of 28px per side (Qwen patch requirement).
    Smaller crops are upscaled proportionally.
    """
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
            Image.LANCZOS
        )
    return crop


class QwenInferenceModule:
    """
    Loads the frozen Qwen2.5-VL model and uses it to classify shelf bounding boxes.

    For each bounding box, the model receives TWO images:
      1. Full image resized to max 1280px (global context)
      2. Crop of the bounding box with 17.5% padding (local detail)

    In case of an OOM error, it automatically falls back to crop-only mode with an adapted prompt.
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

        tok = self.processor.tokenizer
        yes_ids = tok.encode("yes", add_special_tokens=False)
        no_ids  = tok.encode("no",  add_special_tokens=False)
        self.yes_id = yes_ids[0] if yes_ids else None
        self.no_id  = no_ids[0]  if no_ids  else None
        print(f"  [Qwen] yes_token_id={self.yes_id}  no_token_id={self.no_id}")

    def _forward(self, images: list[Image.Image], prompt_text: str,
                 system_prompt: str) -> float:
        """
        Executes a single, localized vision-language forward pass to compute a calibrated 
        probability for the affirmative 'yes' token.

        QUERY EXECUTION & MESSAGE STRUCTURE:
        This function performs exactly ONE query (forward pass) per candidate bounding box. 
        The 'messages' payload is structured as a standard multi-turn chat interaction 
        composed of two roles:
          1. 'system': Contains the core operational instructions and visual inspection criteria.
          2. 'user': Contains a dynamically built list that alternates based on the input. 
             If dual-images are provided, it appends a placeholder for the full image, a placeholder 
             for the target crop, and finally the prompt text. If a fallback occurs, it appends only 
             the single crop placeholder and the adapted prompt text.

        Instead of generating tokens autoregressively, this single pass isolates the logit 
        distribution at the final input position and calculates a targeted binary softmax 
        between the specific token IDs for 'yes' and 'no'.
        """
        n_imgs = len(images)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":
                [{"type": "image"} for _ in range(n_imgs)] +
                [{"type": "text", "text": prompt_text}]
            },
        ]
        text   = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt"
        ).to(self.device)

        out         = self.model(**inputs)
        logits_last = out.logits[0, -1, :]
        yes_l = logits_last[self.yes_id].item() if self.yes_id is not None else 0.0
        no_l  = logits_last[self.no_id].item()  if self.no_id  is not None else 0.0
        return torch.softmax(torch.tensor([yes_l, no_l]), dim=0)[0].item()

    @torch.no_grad()
    def score_boxes(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        prompt_name: str = "no_context",
    ) -> list[float]:
        """
        Returns prob_empty in [0,1] for each bounding box in boxes.

        In case of an OOM error on a single box, it automatically falls back
        to crop-only mode with an adapted prompt and clears the CUDA cache.
        """
        if not boxes:
            return []

        prompt_text       = PROMPTS[prompt_name]
        prompt_text_crop  = PROMPTS[f"{prompt_name}_crop"]
        probs_empty       = []
        oom_count         = 0

        for box in boxes:
            annotated = _draw_target_box(image, box)
            crop      = _extract_crop(image, box)

            try:
                prob_yes = self._forward(
                    images        = [annotated, crop],
                    prompt_text   = prompt_text,
                    system_prompt = SYSTEM_PROMPT,
                )

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                torch.cuda.empty_cache()
                print(f"  [OOM] Fallback to crop-only mode for box {box} "
                      f"(Total OOM occurrences: {oom_count})")
                try:
                    prob_yes = self._forward(
                        images        = [crop],
                        prompt_text   = prompt_text_crop,
                        system_prompt = SYSTEM_PROMPT_CROP_ONLY,
                    )
                except torch.cuda.OutOfMemoryError:
                    # Extreme scenario: even the standalone crop hits an OOM error
                    torch.cuda.empty_cache()
                    print(f"  [OOM] Standalone crop also encountered OOM — forcing prob_empty=0.0")
                    prob_yes = 0.0

            probs_empty.append(prob_yes)

        if oom_count > 0:
            print(f"  [OOM] Total OOM episodes during this batch: {oom_count}")

        return probs_empty
    