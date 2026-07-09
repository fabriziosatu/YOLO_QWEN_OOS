"""
evaluate_yolo_v2.py
====================
Valuta i modelli YOLO sul dataset_finale.

Metriche calcolate:
  - Precision, Recall, F1  (da Ultralytics, IoU=0.5)
  - mAP@0.5, mAP@0.5-95    (da Ultralytics)
  - FP-Rate = FP / (FP + TP)  a IoU: 0.0, 0.25, 0.5, 0.75
  - FN-Rate = FN / (FN + TP)  a IoU: 0.0, 0.25, 0.5, 0.75
  - FPPI    = FP / n_immagini  (a conf=0.25, IoU=0.5)
  - Loss di training (hard-coded dai log)

Uso:
  python evaluate_yolo_v2.py
  python evaluate_yolo_v2.py --data_dir data/dataset_finale
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

CONF_THRESHOLD = 0.25
IOU_THRESHOLDS = [0.0, 0.25, 0.5, 0.75]
IMG_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Dataset ───────────────────────────────────────────────────────────────────

class YOLOFolderDataset(Dataset):
    """Legge immagini e label da <root>/<split>/images/ e <root>/<split>/labels/"""

    def __init__(self, root: str, split: str, img_size: int = 640):
        self.img_size = img_size
        img_dir = Path(root) / split / "images"
        lbl_dir = Path(root) / split / "labels"

        if not img_dir.exists():
            raise FileNotFoundError(f"Cartella immagini non trovata: {img_dir}")

        self.samples = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            self.samples.append((img_path, lbl_path))

        if not self.samples:
            raise RuntimeError(f"Nessuna immagine trovata in {img_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = TF.to_tensor(img)

        boxes = []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append([float(x) for x in parts[1:5]])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) \
                       if boxes else torch.zeros((0, 4))
        return {"image": img_tensor, "boxes": boxes_tensor}


def collate_fn(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "boxes":  [b["boxes"] for b in batch],
    }


# ── Ultralytics val ───────────────────────────────────────────────────────────

def run_ultralytics_val(weights_path: str, data_yaml: str, split: str) -> dict:
    from ultralytics import YOLO
    model  = YOLO(weights_path)
    result = model.val(data=data_yaml, split=split, verbose=False)
    rd     = result.results_dict
    return {
        "Precision":  rd.get("metrics/precision(B)", 0),
        "Recall":     rd.get("metrics/recall(B)",    0),
        "mAP@0.5":    rd.get("metrics/mAP50(B)",     0),
        "mAP@0.5-95": rd.get("metrics/mAP50-95(B)", 0),
    }


# ── Matching predizioni / GT a una data soglia IoU ───────────────────────────

def match_at_iou(pred_boxes, gt_boxes_xyxy, iou_thr: float):
    """
    Associa predizioni a GT con IoU >= iou_thr (greedy matching).

    Nota: con iou_thr=0.0 ogni predizione vicina a una GT viene contata come TP.
    Returns: (tp, fp, fn)
    """
    N = len(pred_boxes)
    M = len(gt_boxes_xyxy)

    if N == 0 and M == 0:
        return 0, 0, 0
    if N == 0:
        return 0, 0, M
    if M == 0:
        return 0, N, 0

    iou_mat = box_iou(pred_boxes, gt_boxes_xyxy)  # (N, M)
    matched = set()
    tp = 0

    for pi in range(N):
        best_iou, best_gt = iou_mat[pi].max(dim=0)
        bg = best_gt.item()
        # Con iou_thr=0.0 accettiamo qualsiasi sovrapposizione > 0
        threshold = max(iou_thr, 1e-6) if iou_thr == 0.0 else iou_thr
        if best_iou.item() >= threshold and bg not in matched:
            tp += 1
            matched.add(bg)

    fp = N - tp
    fn = M - tp
    return tp, fp, fn


def gt_to_xyxy(gt_boxes, img_size):
    """Converte GT da xywh normalizzato a xyxy pixel."""
    s  = float(img_size)
    xc = gt_boxes[:, 0] * s;  yc = gt_boxes[:, 1] * s
    w  = gt_boxes[:, 2] * s;  h  = gt_boxes[:, 3] * s
    return torch.stack([xc - w/2, yc - h/2, xc + w/2, yc + h/2], dim=1)


# ── Calcolo metriche per IoU multipli ────────────────────────────────────────

def compute_metrics_multi_iou(
    weights_path: str,
    data_dir:     str,
    split:        str,
    conf_thr:     float,
    batch_size:   int,
    img_size:     int,
    iou_thresholds: list,
) -> dict:
    """
    Calcola FP-Rate e FN-Rate a diverse soglie IoU.

    FP-Rate = FP / (FP + TP)   — quante predizioni sono sbagliate
    FN-Rate = FN / (FN + TP)   — quanti spazi vuoti reali non vengono trovati
    FPPI    = FP / n_immagini  — falsi positivi medi per immagine (solo a IoU=0.5)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from ultralytics import YOLO as UltralyticsYOLO
    yolo_model = UltralyticsYOLO(weights_path)
    yolo_model.model.eval()

    dataset = YOLOFolderDataset(data_dir, split, img_size=img_size)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, collate_fn=collate_fn)

    # Contatori per ogni soglia IoU
    counters = {iou: {"tp": 0, "fp": 0, "fn": 0} for iou in iou_thresholds}
    n_images = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  conf={conf_thr:.2f} [{split}]",
                          leave=False):
            images = batch["images"].to(device)
            boxes  = batch["boxes"]
            img_sz = images.shape[-1]
            n_images += len(images)

            results = yolo_model.predict(
                images, conf=conf_thr, verbose=False, device=device
            )

            for i, res in enumerate(results):
                gt = boxes[i].to(device)
                gt_xyxy = gt_to_xyxy(gt, img_sz) if len(gt) > 0 \
                          else torch.zeros(0, 4, device=device)

                pred_xyxy = res.boxes.xyxy if len(res.boxes) > 0 \
                            else torch.zeros(0, 4, device=device)

                for iou_thr in iou_thresholds:
                    tp, fp, fn = match_at_iou(pred_xyxy, gt_xyxy, iou_thr)
                    counters[iou_thr]["tp"] += tp
                    counters[iou_thr]["fp"] += fp
                    counters[iou_thr]["fn"] += fn

    # Calcola metriche finali per ogni soglia IoU
    results_by_iou = {}
    for iou_thr, c in counters.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        fp_rate = fp / (fp + tp + 1e-9)   # FP / (FP + TP)
        fn_rate = fn / (fn + tp + 1e-9)   # FN / (FN + TP)
        fppi    = fp / (n_images + 1e-9)  # FP / n_immagini

        results_by_iou[iou_thr] = {
            "tp":      tp,
            "fp":      fp,
            "fn":      fn,
            "fp_rate": fp_rate,
            "fn_rate": fn_rate,
            "fppi":    fppi,
        }

    return {"by_iou": results_by_iou, "n_images": n_images}


# ── Formattazione tabella ─────────────────────────────────────────────────────

def fmt(v, decimals=3):
    return "—" if v is None else f"{v:.{decimals}f}"


def print_main_table(rows: list):
    """Tabella principale: Precision, Recall, F1, mAP, losses."""
    cols = [
        "Run", "Split", "Precision", "Recall", "F1",
        "mAP@0.5", "mAP@0.5-95", "FPPI",
        "box_loss", "cls_loss", "dfl_loss", "tot_loss", "Imgs",
    ]
    widths = {c: max(len(c), 9) for c in cols}
    widths["Run"]   = 16
    widths["Split"] = 6

    header = " | ".join(f"{c:{widths[c]}s}" for c in cols)
    sep    = "-+-".join("-" * widths[c] for c in cols)
    print(f"\n{header}\n{sep}")

    best = {}
    for col in ["Precision", "Recall", "F1", "mAP@0.5", "mAP@0.5-95"]:
        vals = [r[col] for r in rows if r.get(col) is not None]
        best[col] = max(vals) if vals else None
    for col in ["FPPI"]:
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
                if val is not None and col in best and best[col] is not None:
                    if abs(val - best[col]) < 1e-6:
                        s = f"*{s}*"
                cells.append(f"{s:{widths[col]}s}")
        print(" | ".join(cells))
    print("\n  * = miglior valore per colonna")


def print_iou_table(rows: list):
    """Tabella FP-Rate e FN-Rate per ogni soglia IoU."""
    iou_labels = ["0.0", "0.25", "0.5", "0.75"]

    print(f"\n  FP-Rate = FP / (FP + TP)   —   FN-Rate = FN / (FN + TP)")
    print(f"  conf threshold = {CONF_THRESHOLD}\n")

    # Header
    header = f"  {'Run':16s}  {'Split':6s}"
    for iou in iou_labels:
        header += f"  {'FP@'+iou:9s}  {'FN@'+iou:9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    # Trova miglior valore per colonna
    best_fp = {iou: None for iou in iou_labels}
    best_fn = {iou: None for iou in iou_labels}
    for row in rows:
        for iou in iou_labels:
            fp = row.get(f"fp_rate_{iou}")
            fn = row.get(f"fn_rate_{iou}")
            if fp is not None:
                best_fp[iou] = min(best_fp[iou], fp) if best_fp[iou] else fp
            if fn is not None:
                best_fn[iou] = min(best_fn[iou], fn) if best_fn[iou] else fn

    for row in rows:
        line = f"  {row['Run']:16s}  {row['Split']:6s}"
        for iou in iou_labels:
            fp = row.get(f"fp_rate_{iou}")
            fn = row.get(f"fn_rate_{iou}")
            fp_s = fmt(fp)
            fn_s = fmt(fn)
            if fp is not None and best_fp[iou] is not None and abs(fp - best_fp[iou]) < 1e-6:
                fp_s = f"*{fp_s}*"
            if fn is not None and best_fn[iou] is not None and abs(fn - best_fn[iou]) < 1e-6:
                fn_s = f"*{fn_s}*"
            line += f"  {fp_s:9s}  {fn_s:9s}"
        print(line)

    print("\n  * = miglior valore per colonna (il minore)")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/dataset_finale")
    parser.add_argument("--output_dir", default="output/evaluation_table")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size",   type=int, default=640)
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir  = args.data_dir
    data_yaml = str(Path(data_dir) / "data.yaml")
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Struttura dataset:")
    for split in ("train", "val", "test"):
        p = Path(data_dir) / split / "images"
        print(f"    {split}/images  {'✓' if p.exists() else '✗ MANCANTE'}  ({p})")
    print()

    models = {
        "final_noaug": "weights/cluster_training/phase1_yolo_final/weights/best.pt",
        "final_aug":   "weights/cluster_training/phase1_yolo_final_aug/weights/best.pt",
    }

    train_losses = {
        "final_noaug": {"box_loss": 0.6352, "cls_loss": 0.4091,
                        "dfl_loss": 0.0345, "tot_loss": 1.0788},
        "final_aug":   {"box_loss": 0.7258, "cls_loss": 0.5419,
                        "dfl_loss": 0.0314, "tot_loss": 1.2992},
    }

    print("=" * 80)
    print("  VALUTAZIONE — final_noaug vs final_aug")
    print("=" * 80)

    all_rows = []

    for run_name, weights_path in models.items():
        if not Path(weights_path).exists():
            print(f"\n  ⚠  Pesi non trovati: {weights_path} — saltato")
            continue

        print(f"\n{'─'*80}")
        print(f"  Modello : {run_name}")
        print(f"  Pesi    : {weights_path}")

        for split in ("val", "test"):
            print(f"\n  [{split}] Ultralytics val...")
            ul = run_ultralytics_val(weights_path, data_yaml, split)
            p  = ul["Precision"]
            r  = ul["Recall"]
            f1 = 2 * p * r / (p + r + 1e-9)

            print(f"  [{split}] FP-Rate / FN-Rate a IoU multipli (conf={CONF_THRESHOLD})...")
            multi = compute_metrics_multi_iou(
                weights_path, data_dir, split,
                conf_thr=CONF_THRESHOLD,
                batch_size=args.batch_size,
                img_size=args.img_size,
                iou_thresholds=IOU_THRESHOLDS,
            )

            tl = train_losses.get(run_name, {})
            row = {
                "Run":        run_name,
                "Split":      split,
                "Precision":  p,
                "Recall":     r,
                "F1":         f1,
                "mAP@0.5":    ul["mAP@0.5"],
                "mAP@0.5-95": ul["mAP@0.5-95"],
                "FPPI":       multi["by_iou"][0.5]["fppi"],
                "box_loss":   tl.get("box_loss"),
                "cls_loss":   tl.get("cls_loss"),
                "dfl_loss":   tl.get("dfl_loss"),
                "tot_loss":   tl.get("tot_loss"),
                "Imgs":       multi["n_images"],
            }

            # Aggiunge FP-Rate e FN-Rate per ogni soglia IoU
            for iou_thr in IOU_THRESHOLDS:
                key   = str(iou_thr)
                stats = multi["by_iou"][iou_thr]
                row[f"fp_rate_{key}"] = stats["fp_rate"]
                row[f"fn_rate_{key}"] = stats["fn_rate"]
                row[f"tp_{key}"]      = stats["tp"]
                row[f"fp_{key}"]      = stats["fp"]
                row[f"fn_{key}"]      = stats["fn"]

            all_rows.append(row)

            print(f"    Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}  "
                  f"mAP@0.5={ul['mAP@0.5']:.3f}")
            for iou_thr in IOU_THRESHOLDS:
                s = multi["by_iou"][iou_thr]
                print(f"    IoU={iou_thr:.2f}: "
                      f"TP={s['tp']:4d}  FP={s['fp']:4d}  FN={s['fn']:4d}  "
                      f"FP-Rate={s['fp_rate']:.3f}  FN-Rate={s['fn_rate']:.3f}  "
                      f"FPPI={s['fppi']:.3f}")

    if not all_rows:
        print("Nessun modello valutato.")
        return

    # ── Tabelle ───────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  TABELLA PRINCIPALE")
    print(f"{'='*80}")
    print_main_table(all_rows)

    print(f"\n{'='*80}")
    print("  TABELLA FP-Rate / FN-Rate PER SOGLIA IoU")
    print(f"{'='*80}")
    print_iou_table(all_rows)

    # ── Salva JSON e CSV ──────────────────────────────────────────────────────
    json_path = out_dir / "results_table_v2.json"
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2,
                  default=lambda x: None if x != x else x)
    print(f"\n  JSON salvato in : {json_path}")

    cols_csv = [
        "Run", "Split", "Precision", "Recall", "F1",
        "mAP@0.5", "mAP@0.5-95", "FPPI",
        "fp_rate_0.0",  "fn_rate_0.0",
        "fp_rate_0.25", "fn_rate_0.25",
        "fp_rate_0.5",  "fn_rate_0.5",
        "fp_rate_0.75", "fn_rate_0.75",
        "box_loss", "cls_loss", "dfl_loss", "tot_loss", "Imgs",
    ]
    csv_path = out_dir / "results_table_v2.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(cols_csv) + "\n")
        for row in all_rows:
            f.write(",".join(
                str(row.get(c, "")) if row.get(c) is not None else ""
                for c in cols_csv
            ) + "\n")
    print(f"  CSV salvato in  : {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()