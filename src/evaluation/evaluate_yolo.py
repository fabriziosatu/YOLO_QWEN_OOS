"""
scripts/evaluate_table.py
==========================
Valuta i modelli YOLO sul dataset merged e produce una tabella
con le stesse colonne della tabella di riferimento della collega.

Modelli valutati:
  - phase1_yolo_merged      (No Data Aug)
  - phase1_yolo_merged_aug  (Data Aug)

Per ogni modello valuta su val e test set.

Colonne prodotte:
  Run, Split, Precision, Recall, F1, mAP@0.5, mAP@0.5-95,
  P@0.5, P@0.0, FP-Rate, box_loss, cls_loss, dfl_loss, tot_loss

Uso:
  python scripts/evaluate_table.py
  python scripts/evaluate_table.py --data_dir data/merged_dataset
"""

import argparse
import sys
import json
import torch
import numpy as np
from pathlib import Path
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, ".")

from src.data.dataset import build_dataloaders
from src.utils.config import CFG


IOU_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/merged_dataset")
    parser.add_argument("--output_dir", default="output/evaluation_table")
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


# ── Ultralytics val (Precision, Recall, mAP, losses) ─────────────────────────

def run_ultralytics_val(weights_path: str, data_yaml: str, split: str) -> dict:
    """
    Usa Ultralytics nativo per ottenere mAP, losses e metriche ufficiali.
    """
    from ultralytics import YOLO
    model  = YOLO(weights_path)
    result = model.val(data=data_yaml, split=split, verbose=False)
    rd     = result.results_dict

    # Loss disponibili solo su val (non su test)
    box_loss = rd.get("val/box_loss", None)
    cls_loss = rd.get("val/cls_loss", None)
    dfl_loss = rd.get("val/dfl_loss", None)
    tot_loss = (box_loss + cls_loss + dfl_loss) if all(
        v is not None for v in [box_loss, cls_loss, dfl_loss]
    ) else None

    return {
        "Precision": rd.get("metrics/precision(B)", 0),
        "Recall":    rd.get("metrics/recall(B)",    0),
        "mAP@0.5":   rd.get("metrics/mAP50(B)",     0),
        "mAP@0.5-95":rd.get("metrics/mAP50-95(B)",  0),
        "box_loss":  box_loss,
        "cls_loss":  cls_loss,
        "dfl_loss":  dfl_loss,
        "tot_loss":  tot_loss,
    }


# ── FP-Rate e P@conf custom ───────────────────────────────────────────────────

def match_preds_gt(pred_boxes, gt_boxes, img_size, iou_thr=IOU_THRESHOLD):
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    s  = float(img_size)
    xc = gt_boxes[:, 0]*s; yc = gt_boxes[:, 1]*s
    w  = gt_boxes[:, 2]*s; h  = gt_boxes[:, 3]*s
    gt_xyxy = torch.stack([xc-w/2, yc-h/2, xc+w/2, yc+h/2], dim=1)

    iou_mat   = box_iou(pred_boxes, gt_xyxy)
    matched   = set()
    tp        = 0
    for pi in range(len(pred_boxes)):
        best_iou, best_gt = iou_mat[pi].max(dim=0)
        bg = best_gt.item()
        if best_iou.item() >= iou_thr and bg not in matched:
            tp += 1
            matched.add(bg)
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp


def compute_fp_rate_and_precision(
    weights_path: str,
    data_dir:     str,
    split:        str,
    conf_thr:     float,
    batch_size:   int,
) -> dict:
    """
    Calcola Precision, FP-Rate e FPPI a una data soglia di confidenza.

    FPPI (False Positives Per Image):
        FPPI = totale_FP / totale_immagini
        Misura quanti falsi positivi produce il modello in media per immagine.
        Non richiede TN — e' una metrica pulita e standard in contesti industriali.
        Es: FPPI=0.2 significa che in media il modello sbaglia su 1 immagine ogni 5.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from ultralytics import YOLO as UltralyticsYOLO
    yolo_model = UltralyticsYOLO(weights_path)
    yolo_model.model.eval()

    _, val_loader, test_loader = build_dataloaders(
        img_size   = CFG.data.img_size,
        batch_size = batch_size,
        mode       = "phase1",
        data_dir   = data_dir,
    )
    loader = val_loader if split == "val" else test_loader

    total_tp = total_fp = total_fn = total_tn = 0
    n_images_total = 0   # tutte le immagini, con o senza predizioni

    with torch.no_grad():
        for batch in tqdm(loader,
                          desc=f"  FPPI/FP-Rate conf={conf_thr} [{split}]",
                          leave=False):
            images = batch["images"].to(device)
            boxes  = batch["boxes"]
            img_sz = images.shape[-1]

            results = yolo_model.predict(
                images, conf=conf_thr, verbose=False, device=device
            )

            for i, res in enumerate(results):
                gt = boxes[i].to(device)
                n_images_total += 1   # conta SEMPRE ogni immagine

                if len(res.boxes) == 0:
                    total_fn += len(gt)
                    total_tn += 1
                    continue

                pred_xyxy = res.boxes.xyxy
                tp, fp, fn = match_preds_gt(pred_xyxy, gt, img_sz)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                if fp == 0:
                    total_tn += 1

    precision = total_tp / (total_tp + total_fp + 1e-9)
    fp_rate   = total_fp / (total_fp + total_tn + 1e-9)
    fppi      = total_fp / (n_images_total + 1e-9)   # FP / n_immagini totali

    return {
        "precision":    precision,
        "fp_rate":      fp_rate,
        "fppi":         fppi,
        "n_images":     n_images_total,
        "tp": total_tp, "fp": total_fp,
        "fn": total_fn, "tn": total_tn,
    }


# ── Formattazione tabella ─────────────────────────────────────────────────────

def fmt(v, decimals=3):
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"


def print_table(rows: list):
    """Stampa la tabella in formato leggibile."""
    cols = [
        "Run", "Split", "Precision", "Recall", "F1",
        "mAP@0.5", "mAP@0.5-95", "P@0.5", "P@0.0", "FP-Rate", "FPPI",
        "box_loss", "cls_loss", "dfl_loss", "tot_loss", "Imgs"
    ]
    widths = {c: max(len(c), 9) for c in cols}
    widths["Run"]   = 28
    widths["Split"] = 6

    header = " | ".join(f"{c:{widths[c]}s}" for c in cols)
    sep    = "-+-".join("-" * widths[c] for c in cols)

    print(f"\n{header}")
    print(sep)

    # Trova i migliori valori per evidenziarli
    metric_cols = ["Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5-95",
                   "P@0.5", "P@0.0"]
    best = {}
    for col in metric_cols:
        vals = [r[col] for r in rows if r[col] is not None]
        best[col] = max(vals) if vals else None
    # FP-Rate e FPPI: il minore è il migliore
    for col in ("FP-Rate", "FPPI"):
        vals = [r[col] for r in rows if r.get(col) is not None]
        best[col] = min(vals) if vals else None

    for row in rows:
        cells = []
        for col in cols:
            val = row.get(col)
            if col in ("Run", "Split"):
                cells.append(f"{str(val):{widths[col]}s}")
            else:
                s = fmt(val)
                # Evidenzia il miglior valore
                if val is not None and col in best and best[col] is not None:
                    is_best = (
                        (col != "FP-Rate" and abs(val - best[col]) < 1e-6) or
                        (col == "FP-Rate" and abs(val - best[col]) < 1e-6)
                    )
                    if is_best:
                        s = f"*{s}*"
                cells.append(f"{s:{widths[col]}s}")
        print(" | ".join(cells))

    print(f"\n  * = miglior valore per colonna")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    data_dir  = args.data_dir
    data_yaml = str(Path(data_dir) / "data.yaml")
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "final_noaug": "weights/cluster_training/phase1_yolo_final/weights/best.pt",
        "final_aug":   "weights/cluster_training/phase1_yolo_final_aug/weights/best.pt",
    }

    train_losses = {
    "final_noaug": {"box_loss": 0.6352, "cls_loss": 0.4091, "dfl_loss": 0.0345, "tot_loss": 1.0788},
    "final_aug":   {"box_loss": 0.7258, "cls_loss": 0.5419, "dfl_loss": 0.0314, "tot_loss": 1.2992},
    }

    print("=" * 80)
    print("  VALUTAZIONE COMPARATIVA — final_noaug vs final_aug")
    print("=" * 80)

    all_rows = []

    for run_name, weights_path in models.items():
        if not Path(weights_path).exists():
            print(f"\n  ⚠ Pesi non trovati: {weights_path} — saltato")
            continue

        print(f"\n{'─'*80}")
        print(f"  Modello: {run_name}")
        print(f"  Pesi   : {weights_path}")

        for split in ("val", "test"):
            print(f"\n  [{split}] Ultralytics val...")

            # Metriche Ultralytics (mAP, losses)
            ul = run_ultralytics_val(weights_path, data_yaml, split)

            # F1
            p  = ul["Precision"]
            r  = ul["Recall"]
            f1 = 2*p*r/(p+r+1e-9)

            # P@0.5 e FP-Rate con conf=0.25
            print(f"  [{split}] FP-Rate con conf=0.25...")
            m25 = compute_fp_rate_and_precision(
                weights_path, data_dir, split,
                conf_thr=0.25, batch_size=args.batch_size
            )

            # P@0.0 e FP-Rate con conf=0.0
            print(f"  [{split}] FP-Rate con conf=0.0...")
            m00 = compute_fp_rate_and_precision(
                weights_path, data_dir, split,
                conf_thr=0.0, batch_size=1
            )

            row = {
                "Run":        run_name,
                "Split":      split,
                "Precision":  p,
                "Recall":     r,
                "F1":         f1,
                "mAP@0.5":    ul["mAP@0.5"],
                "mAP@0.5-95": ul["mAP@0.5-95"],
                "P@0.5":      m25["precision"],
                "P@0.0":      m00["precision"],
                "FP-Rate":    m25["fp_rate"],
                "FPPI":       m25["fppi"],
                "box_loss":   train_losses.get(run_name, {}).get("box_loss"),
                "cls_loss":   train_losses.get(run_name, {}).get("cls_loss"),
                "dfl_loss":   train_losses.get(run_name, {}).get("dfl_loss"),
                "tot_loss":   train_losses.get(run_name, {}).get("tot_loss"),
                "Imgs":       m25["n_images"],
            }
            all_rows.append(row)

            print(f"    Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}  "
                  f"mAP@0.5={ul['mAP@0.5']:.3f}")
            print(f"    P@0.5={m25['precision']:.3f}  "
                  f"P@0.0={m00['precision']:.3f}  "
                  f"FP-Rate={m25['fp_rate']:.3f}  "
                  f"FPPI={m25['fppi']:.3f}")

    # ── Stampa tabella finale ─────────────────────────────────────────────────
    print_table(all_rows)

    # ── Salva JSON e CSV ──────────────────────────────────────────────────────
    json_path = out_dir / "results_table.json"
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=lambda x: None if x != x else x)
    print(f"\n  Risultati salvati in: {json_path}")

    # CSV per Excel/Sheets
    csv_path = out_dir / "results_table.csv"
    cols = ["Run", "Split", "Precision", "Recall", "F1",
            "mAP@0.5", "mAP@0.5-95", "P@0.5", "P@0.0", "FP-Rate", "FPPI",
            "box_loss", "cls_loss", "dfl_loss", "tot_loss", "Imgs"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for row in all_rows:
            f.write(",".join(
                str(row.get(c, "")) if row.get(c) is not None else ""
                for c in cols
            ) + "\n")
    print(f"  CSV per Excel     : {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()