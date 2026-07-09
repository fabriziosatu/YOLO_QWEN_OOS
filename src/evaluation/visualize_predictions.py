"""
scripts/visualize_predictions.py
==================================
Visualizza le predizioni della pipeline joint su immagini del test set.

Legenda:
  Giallo tratteggiato = GT box reale
  Blu                 = predizione YOLO
  Verde               = ROI classificata VUOTA da SigLIP (MLP)
  Rosso               = ROI classificata PIENA da SigLIP (MLP)

Esegui con:
  python scripts/visualize_predictions.py
  python scripts/visualize_predictions.py --n_images 8
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

sys.path.insert(0, ".")

from src.data.dataset          import build_dataloaders
from src.models.joint_pipeline import JointPipeline
from src.utils.config          import CFG, PHASE1_WEIGHTS, PHASE2_WEIGHTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1_weights", type=str, default=str(PHASE1_WEIGHTS))
    parser.add_argument("--phase2_weights", type=str, default=str(PHASE2_WEIGHTS))
    parser.add_argument("--data_dir",      type=str,   default="data/processed_clean")
    parser.add_argument("--n_images",      type=int,   default=8)
    parser.add_argument("--output_dir",    type=str,   default="output/visualizations")
    parser.add_argument("--conf",          type=float, default=0.15)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--siglip_thresh", type=float, default=0.5)
    return parser.parse_args()


def draw_box(draw, box, color, label="", width=2, dashed=False):
    x1, y1, x2, y2 = [int(v) for v in box]
    if dashed:
        dash = 8
        for x in range(x1, x2, dash*2):
            draw.line([(x, y1), (min(x+dash, x2), y1)], fill=color, width=width)
            draw.line([(x, y2), (min(x+dash, x2), y2)], fill=color, width=width)
        for y in range(y1, y2, dash*2):
            draw.line([(x1, y), (x1, min(y+dash, y2))], fill=color, width=width)
            draw.line([(x2, y), (x2, min(y+dash, y2))], fill=color, width=width)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        bbox_text = draw.textbbox((x1, y1-16), label, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((x1, y1-16), label, fill="white", font=font)


def visualize_single(image_tensor, gt_boxes, output, conf_threshold,
                     siglip_thresh=0.5, img_path=""):
    img_np = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img    = Image.fromarray(img_np).convert("RGB")
    draw   = ImageDraw.Draw(img)
    W, H   = img.size

    # GT box (giallo tratteggiato)
    for gt in gt_boxes:
        xc, yc, w, h = gt[0]*W, gt[1]*H, gt[2]*W, gt[3]*H
        x1, y1 = xc - w/2, yc - h/2
        x2, y2 = xc + w/2, yc + h/2
        draw_box(draw, [x1, y1, x2, y2], color=(255, 255, 0),
                 label="GT", dashed=True, width=2)

    # Predizioni YOLO (blu)
    raw_preds = output["predictions"]
    preds = raw_preds[0] if raw_preds.dim() == 3 else raw_preds
    for pred in preds:
        if pred[4].item() >= conf_threshold:
            x1, y1, x2, y2 = pred[:4].tolist()
            conf = pred[4].item()
            draw_box(draw, [x1, y1, x2, y2],
                     color=(100, 100, 255),
                     label=f"YOLO {conf:.2f}",
                     width=1)

    # ROI classificate da SigLIP MLP — usa logits_pos
    if output["n_rois"] > 0:
        rois   = output["rois"]
        logits = output["logits_pos"]   # (N, 1) — FIX: era output["logits"]

        for j in range(len(rois)):
            if rois[j, 0].long() != 0:   # solo img 0 del batch
                continue
            x1, y1, x2, y2 = rois[j, 1:].tolist()
            score    = torch.sigmoid(logits[j]).item()
            is_empty = score >= siglip_thresh
            color    = (0, 220, 0) if is_empty else (220, 0, 0)
            label    = f"VUOTO {score:.2f}" if is_empty else f"PIENO {score:.2f}"
            draw_box(draw, [x1, y1, x2, y2], color=color, label=label, width=3)

    # Legenda
    legend_items = [
        ((255, 255, 0),   "GT box reale"),
        ((100, 100, 255), "Predizione YOLO"),
        ((0, 220, 0),     f"SigLIP: VUOTO (>={siglip_thresh:.2f})"),
        ((220, 0, 0),     f"SigLIP: PIENO (<{siglip_thresh:.2f})"),
    ]
    y_leg = 5
    for color, text in legend_items:
        draw.rectangle([5, y_leg, 20, y_leg+14], fill=color, outline="white")
        draw.text((25, y_leg), text, fill="white")
        y_leg += 18

    if img_path:
        draw.text((W-200, 5), Path(img_path).name, fill="white")

    return img


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  VISUALIZZAZIONE PREDIZIONI PIPELINE JOINT")
    print("=" * 60)
    print(f"  Fase 1 weights  : {args.phase1_weights}")
    print(f"  Fase 2 weights  : {args.phase2_weights}")
    print(f"  N immagini      : {args.n_images}")
    print(f"  SigLIP thresh   : {args.siglip_thresh}")
    print(f"  Output          : {out_dir}")
    print("-" * 60)

    # ── Carica pipeline ───────────────────────────────────────────────────────
    # FIX: rimossi mlp_hidden e mlp_dropout — parametri di SigLIPModule,
    # non di JointPipeline
    pipeline = JointPipeline(
        yolo_weights      = args.phase1_weights,
        siglip_model_name = CFG.siglip.model_name,
        conf_threshold    = args.conf,
        lora_r_visual     = CFG.siglip.lora_r_visual,
        lora_alpha_visual = CFG.siglip.lora_alpha_visual,
        lora_dropout      = CFG.siglip.lora_dropout,
        roi_size          = CFG.data.roi_size,
    ).to(device)

    ck    = torch.load(args.phase2_weights, map_location=device, weights_only=False)
    state = {k: v for k, v in ck["pipeline_state_dict"].items()
             if "model.23" not in k}
    pipeline.load_state_dict(state, strict=False)
    pipeline.eval()
    print(f"  Pipeline caricata — epoca {ck['epoch']+1}")

    # ── Test set — phase1 per avere solo positivi con GT ─────────────────────
    _, _, test_loader = build_dataloaders(
        img_size   = CFG.data.img_size,
        batch_size = 1,
        mode       = "phase1",
        data_dir   = args.data_dir,
    )

    all_batches = []
    for i, batch in enumerate(test_loader):
        all_batches.append(batch)
        if i >= 50:
            break
    selected = random.sample(all_batches, min(args.n_images, len(all_batches)))
    print(f"  Visualizzo {len(selected)} immagini → {out_dir}/\n")

    with torch.no_grad():
        for idx, batch in enumerate(selected):
            images = batch["images"].to(device)
            boxes  = batch["boxes"]

            output = pipeline(images, gt_boxes=None)

            gt  = boxes[0]
            viz = visualize_single(
                image_tensor   = images[0],
                gt_boxes       = gt,
                output         = output,
                conf_threshold = args.conf,
                siglip_thresh  = args.siglip_thresh,
            )

            n_rois = output["n_rois"]
            if n_rois > 0:
                scores  = torch.sigmoid(output["logits_pos"]).squeeze(1)
                n_empty = int((scores >= args.siglip_thresh).sum().item())
                n_full  = n_rois - n_empty
            else:
                n_empty = n_full = 0

            fname = out_dir / f"pred_{idx+1:03d}.jpg"
            viz.save(fname)
            print(f"  [{idx+1:2d}] GT={len(gt)} | "
                  f"ROIs={n_rois} (vuoti={n_empty}, pieni={n_full}) | "
                  f"→ {fname.name}")

    print(f"\n  ✓ Salvate {len(selected)} immagini in {out_dir}/")
    print(f"\n  Legenda:")
    print(f"    Giallo tratteggiato = GT box reale")
    print(f"    Blu                 = Predizione YOLO")
    print(f"    Verde               = SigLIP: scaffale VUOTO")
    print(f"    Rosso               = SigLIP: scaffale PIENO")


if __name__ == "__main__":
    main()