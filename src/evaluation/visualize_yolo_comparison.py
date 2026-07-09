"""
scripts/visualize_yolo_comparison.py
======================================
Visualizza predizioni YOLO side-by-side per confrontare
No Augmentation vs Data Augmentation sulle stesse immagini.

Per ogni immagine produce un file affiancato:
  Sinistra: No Aug  |  Destra: Data Aug

Legenda:
  Giallo tratteggiato = GT box reale
  Verde pieno         = True Positive (predizione corretta, IoU >= 0.5)
  Rosso pieno         = False Positive (predizione sbagliata)
  Giallo puntinato    = False Negative (GT non trovata)

Uso:
  python scripts/visualize_yolo_comparison.py
  python scripts/visualize_yolo_comparison.py --n_images 10 --split test
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou
import random

sys.path.insert(0, ".")

from src.utils.config import CFG

IOU_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noaug_weights", type=str,
        default="weights/cluster_training/phase1_yolo_final/weights/best.pt")
    parser.add_argument("--aug_weights", type=str,
        default="weights/cluster_training/phase1_yolo_final_aug/weights/best.pt")
    parser.add_argument("--data_dir",   type=str, default="data/dataset_finale")
    parser.add_argument("--split",      type=str, default="test",
        choices=["val", "test"])
    parser.add_argument("--n_images",   type=int, default=8)
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="output/yolo_comparison")
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_font(size=12):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_box(draw, box, color, label="", width=2, dashed=False):
    x1, y1, x2, y2 = [int(v) for v in box]
    if dashed:
        dash = 6
        for x in range(x1, x2, dash * 2):
            draw.line([(x, y1), (min(x+dash, x2), y1)], fill=color, width=width)
            draw.line([(x, y2), (min(x+dash, x2), y2)], fill=color, width=width)
        for y in range(y1, y2, dash * 2):
            draw.line([(x1, y), (x1, min(y+dash, y2))], fill=color, width=width)
            draw.line([(x2, y), (x2, min(y+dash, y2))], fill=color, width=width)
    else:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label:
        font = get_font(11)
        try:
            bbox_text = draw.textbbox((x1, max(y1-16, 0)), label, font=font)
            draw.rectangle(bbox_text, fill=color)
            draw.text((x1, max(y1-16, 0)), label, fill="white", font=font)
        except Exception:
            draw.text((x1, max(y1-16, 0)), label, fill=color)


def classify_predictions(pred_boxes, gt_boxes_xyxy, iou_thr=IOU_THRESHOLD):
    """
    Classifica ogni predizione come TP o FP e identifica i FN.
    Returns:
        tp_mask : bool tensor (N,) — True se la predizione e' un TP
        fn_mask : bool tensor (M,) — True se la GT non e' stata trovata
    """
    N = len(pred_boxes)
    M = len(gt_boxes_xyxy) if gt_boxes_xyxy is not None else 0

    tp_mask = torch.zeros(N, dtype=torch.bool)
    fn_mask = torch.ones(M,  dtype=torch.bool)  # inizia tutto FN

    if N == 0 or M == 0:
        return tp_mask, fn_mask

    iou_mat  = box_iou(pred_boxes, gt_boxes_xyxy)  # (N, M)
    matched  = set()
    for pi in range(N):
        best_iou, best_gt = iou_mat[pi].max(dim=0)
        bg = best_gt.item()
        if best_iou.item() >= iou_thr and bg not in matched:
            tp_mask[pi] = True
            fn_mask[bg] = False
            matched.add(bg)

    return tp_mask, fn_mask


def gt_to_xyxy(gt_boxes, W, H):
    """Converte GT da xywh norm a xyxy pixel."""
    if len(gt_boxes) == 0:
        return torch.zeros(0, 4)
    xc = gt_boxes[:, 0] * W;  yc = gt_boxes[:, 1] * H
    w  = gt_boxes[:, 2] * W;  h  = gt_boxes[:, 3] * H
    return torch.stack([xc-w/2, yc-h/2, xc+w/2, yc+h/2], dim=1)


def render_single(image_tensor, gt_boxes, pred_boxes_conf, model_label):
    """
    Disegna predizioni su una singola immagine.

    Colori:
      Giallo tratteg. = GT reale
      Verde           = TP (trovata correttamente)
      Rosso           = FP (falso positivo)
      Arancio tratt.  = FN (GT non trovata)
    """
    img_np = (image_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    img    = Image.fromarray(img_np).convert("RGB")
    draw   = ImageDraw.Draw(img)
    W, H   = img.size

    gt_xyxy = gt_to_xyxy(gt_boxes, W, H)

    # Filtra predizioni per confidenza
    if len(pred_boxes_conf) > 0:
        mask       = pred_boxes_conf[:, 4] >= 0.0  # già filtrate
        pred_xyxy  = pred_boxes_conf[:, :4]
        pred_conf  = pred_boxes_conf[:, 4]
    else:
        pred_xyxy = torch.zeros(0, 4)
        pred_conf = torch.zeros(0)

    # Classifica TP/FP/FN
    tp_mask, fn_mask = classify_predictions(pred_xyxy, gt_xyxy)

    # Disegna GT (giallo tratteggiato) — sottofondo
    for i, gt in enumerate(gt_xyxy):
        draw_box(draw, gt.tolist(), color=(255, 220, 0),
                 label="GT", dashed=True, width=2)

    # Disegna FN (arancio tratteggiato) — GT non trovate
    for i, is_fn in enumerate(fn_mask):
        if is_fn:
            draw_box(draw, gt_xyxy[i].tolist(), color=(255, 140, 0),
                     label="FN", dashed=True, width=3)

    # Disegna predizioni — verde=TP, rosso=FP
    for i in range(len(pred_xyxy)):
        conf  = pred_conf[i].item()
        is_tp = tp_mask[i].item()
        color = (0, 200, 0) if is_tp else (220, 0, 0)
        tag   = "TP" if is_tp else "FP"
        draw_box(draw, pred_xyxy[i].tolist(),
                 color=color, label=f"{tag} {conf:.2f}", width=2)

    # Titolo modello
    font_title = get_font(14)
    tp_count = tp_mask.sum().item()
    fp_count = (~tp_mask).sum().item() if len(tp_mask) > 0 else 0
    fn_count = fn_mask.sum().item()
    title = f"{model_label}  |  TP={tp_count}  FP={fp_count}  FN={fn_count}"
    draw.rectangle([0, H-28, W, H], fill=(30, 30, 30))
    draw.text((5, H-24), title, fill="white", font=font_title)

    # Legenda
    legend = [
        ((255, 220, 0), "GT reale"),
        ((0, 200, 0),   "True Positive"),
        ((220, 0, 0),   "False Positive"),
        ((255, 140, 0), "False Negative"),
    ]
    y_leg = 5
    for color, text in legend:
        draw.rectangle([5, y_leg, 18, y_leg+12], fill=color, outline="white")
        draw.text((22, y_leg), text, fill="white", font=get_font(10))
        y_leg += 16

    return img


def make_comparison(img_noaug, img_aug, separator_px=6):
    """Affianca le due immagini con un separatore."""
    W, H = img_noaug.size
    canvas = Image.new("RGB", (W * 2 + separator_px, H), color=(50, 50, 50))
    canvas.paste(img_noaug, (0, 0))
    canvas.paste(img_aug,   (W + separator_px, 0))

    # Etichette in alto
    draw = ImageDraw.Draw(canvas)
    font = get_font(15)
    draw.rectangle([0, 0, W, 22], fill=(30, 30, 60, 200))
    draw.text((8, 4), "NO AUGMENTATION", fill=(200, 200, 255), font=font)
    draw.rectangle([W + separator_px, 0, W * 2 + separator_px, 22],
                   fill=(30, 60, 30, 200))
    draw.text((W + separator_px + 8, 4), "DATA AUGMENTATION",
              fill=(180, 255, 180), font=font)

    return canvas


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  CONFRONTO VISIVO: No Aug vs Data Aug")
    print("=" * 65)
    print(f"  No Aug  : {args.noaug_weights}")
    print(f"  Aug     : {args.aug_weights}")
    print(f"  Split   : {args.split}")
    print(f"  Conf    : {args.conf}")
    print(f"  Output  : {out_dir}")
    print("-" * 65)

    # ── Carica modelli ────────────────────────────────────────────────────────
    from ultralytics import YOLO
    model_noaug = YOLO(args.noaug_weights)
    model_aug   = YOLO(args.aug_weights)
    print("  Modelli caricati.")

    # ── Carica immagini — supporta struttura A (images/split) e B (split/images)
    data_root = Path(args.data_dir)
    split     = args.split
    img_dir = (data_root / "images" / split) if (data_root / "images" / split).exists()               else (data_root / split / "images")
    lbl_dir = (data_root / "labels" / split) if (data_root / "labels" / split).exists()               else (data_root / split / "labels")
    if not img_dir.exists():
        raise RuntimeError(
            f"Cartella immagini non trovata.\n"
            f"  Provato: {data_root / 'images' / split}\n"
            f"  Provato: {data_root / split / 'images'}"
        )
    IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}
    all_img_paths = sorted([p for p in img_dir.glob("*")
                             if p.suffix.lower() in IMG_SUFFIXES])
    if not all_img_paths:
        raise RuntimeError(f"Nessuna immagine trovata in {img_dir}")
    print(f"  Dataset  : {img_dir} ({len(all_img_paths)} immagini)")
    random.shuffle(all_img_paths)
    selected_paths = all_img_paths[:args.n_images]

    from PIL import Image as PILImage
    import torchvision.transforms.functional as TF

    def load_sample(img_path):
        img = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        target = CFG.data.img_size
        scale  = min(target / orig_w, target / orig_h)
        new_w  = int(orig_w * scale); new_h = int(orig_h * scale)
        img    = img.resize((new_w, new_h), PILImage.BILINEAR)
        pad_l  = (target - new_w) // 2; pad_t = (target - new_h) // 2
        canvas = PILImage.new("RGB", (target, target), (114, 114, 114))
        canvas.paste(img, (pad_l, pad_t))
        img_t  = TF.to_tensor(canvas)
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            xc, yc, w, h = (float(p) for p in parts[1:])
                            xc_new = (xc * orig_w * scale + pad_l) / target
                            yc_new = (yc * orig_h * scale + pad_t) / target
                            w_new  = (w  * orig_w * scale)          / target
                            h_new  = (h  * orig_h * scale)          / target
                            boxes.append([xc_new, yc_new, w_new, h_new])
                        except ValueError:
                            pass
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes                   else torch.zeros((0, 4), dtype=torch.float32)
        return img_t, boxes_t

    print(f"  Visualizzo {len(selected_paths)} immagini...\n")

    stats = {"noaug": {"tp":0,"fp":0,"fn":0}, "aug": {"tp":0,"fp":0,"fn":0}}

    with torch.no_grad():
        for idx, img_path in enumerate(selected_paths):
            img_t, boxes = load_sample(img_path)
            images = img_t.unsqueeze(0)   # (1, 3, H, W)
            W = H  = CFG.data.img_size

            gt_xyxy = gt_to_xyxy(boxes, W, H)

            # Predizioni No Aug
            res_noaug = model_noaug.predict(
                images.to(device), conf=args.conf, verbose=False, device=device
            )[0]
            preds_noaug = torch.cat([
                res_noaug.boxes.xyxy,
                res_noaug.boxes.conf.unsqueeze(1)
            ], dim=1).cpu() if len(res_noaug.boxes) > 0 else torch.zeros(0, 5)

            # Predizioni Aug
            res_aug = model_aug.predict(
                images.to(device), conf=args.conf, verbose=False, device=device
            )[0]
            preds_aug = torch.cat([
                res_aug.boxes.xyxy,
                res_aug.boxes.conf.unsqueeze(1)
            ], dim=1).cpu() if len(res_aug.boxes) > 0 else torch.zeros(0, 5)

            # Render
            img_noaug_vis = render_single(img_t, boxes, preds_noaug, "No Aug")
            img_aug_vis   = render_single(img_t, boxes, preds_aug,   "Data Aug")
            comparison    = make_comparison(img_noaug_vis, img_aug_vis)

            # Statistiche
            tp_n, fn_n = classify_predictions(preds_noaug[:, :4], gt_xyxy)
            tp_a, fn_a = classify_predictions(preds_aug[:, :4],   gt_xyxy)
            stats["noaug"]["tp"] += tp_n.sum().item()
            stats["noaug"]["fp"] += (~tp_n).sum().item()
            stats["noaug"]["fn"] += fn_n.sum().item()
            stats["aug"]["tp"]   += tp_a.sum().item()
            stats["aug"]["fp"]   += (~tp_a).sum().item()
            stats["aug"]["fn"]   += fn_a.sum().item()

            fname = out_dir / f"comparison_{idx+1:03d}.jpg"
            comparison.save(fname, quality=92)

            print(f"  [{idx+1:2d}] GT={len(boxes)} | "
                  f"NoAug: TP={tp_n.sum().item()} FP={(~tp_n).sum().item()} FN={fn_n.sum().item()} | "
                  f"Aug:   TP={tp_a.sum().item()} FP={(~tp_a).sum().item()} FN={fn_a.sum().item()} | "
                  f"→ {fname.name}")

    # Riepilogo
    print(f"\n{'='*65}")
    print(f"  RIEPILOGO su {len(selected_paths)} immagini ({args.split})")
    print(f"{'='*65}")
    for name, s in stats.items():
        p = s["tp"] / (s["tp"] + s["fp"] + 1e-9)
        r = s["tp"] / (s["tp"] + s["fn"] + 1e-9)
        f1 = 2*p*r / (p+r+1e-9)
        print(f"  {name.upper():8s}: TP={s['tp']:4d}  FP={s['fp']:4d}  FN={s['fn']:4d}  "
              f"P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    print(f"\n  Immagini salvate in: {out_dir}/")


if __name__ == "__main__":
    main()