"""
evaluate_yolo_qwen.py  —  v3
-----------------------------
Valuta 4 pipeline: 2 detector (noaug / aug) × 2 prompt Qwen (no_context / context)

NOVITÀ v3:
  - Qwen riceve l'immagine INTERA con BB annotate (verde = in esame, rosso = altre)
  - Una query per BB
  - Le BB predette da YOLO vengono classificate come TP o FP rispetto al GT
    PRIMA di passarle a Qwen, così si misura:
      · Su BB che sono TP reali: Qwen dice yes (corretto) o no (errore)?
      · Su BB che sono FP reali: Qwen dice yes (errore) o no (corretto)?
  - Metriche finali: FPR, FNR, TPR + accuracy Qwen su TP e su FP separatamente

Uso:
    python3 evaluate_yolo_qwen.py \
        --noaug_weights /cache/.../best.pt \
        --aug_weights   /cache/.../best_aug.pt \
        --qwen_path     /cache/.../Qwen2.5-VL-3B-Instruct \
        --test_images   /cache/.../test/images \
        --test_labels   /cache/.../test/labels \
        --out_dir       /cache/.../results/yolo_qwen_v3
"""

import argparse, json, time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
CONF_THRESHOLDS  = [0.25, 0.5]
IOU_THRESHOLDS   = [0.0, 0.25, 0.5, 0.75]
QWEN_THRESHOLDS  = [0.3, 0.4, 0.5, 0.6]
PROMPT_NAMES     = ["no_context", "context"]
# ─────────────────────────────────────────────────────────────────────────────


# ── Utility ──────────────────────────────────────────────────────────────────

def load_test_pairs(img_dir: Path, lbl_dir: Path) -> list[dict]:
    """Carica coppie immagine/label. Ogni GT box è [x1,y1,x2,y2] in pixel."""
    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img   = Image.open(img_path).convert("RGB")
        W, H  = img.size
        gt_boxes = []
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            gt_boxes.append([x1, y1, x2, y2])
        pairs.append({"path": img_path, "image": img, "gt": gt_boxes, "W": W, "H": H})
    return pairs


def iou(box_a: list, box_b: list) -> float:
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def classify_pred_boxes(pred_boxes: list, gt_boxes: list, iou_thr: float) -> list[bool]:
    """
    Ritorna lista di bool: True = TP (la pred matcha una GT), False = FP.
    Ogni GT viene matchata al massimo una volta (greedy per confidenza).
    """
    matched_gt = set()
    is_tp = []
    for pb in pred_boxes:
        best_iou = 0.0
        best_gt  = -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            v = iou(pb, gb)
            if v > best_iou:
                best_iou = v
                best_gt  = j
        if best_iou >= iou_thr and best_gt >= 0:
            matched_gt.add(best_gt)
            is_tp.append(True)
        else:
            is_tp.append(False)
    return is_tp


def compute_metrics(all_results: list[dict], iou_thr: float, qwen_thr: float,
                    detector: str, prompt: str) -> dict:
    """
    Calcola FPR, FNR, TPR e accuracy Qwen su TP/FP separati.

    Una BB predetta viene "tenuta" se prob_empty >= qwen_thr.
    TP finale = predizione tenuta E corrispondente GT box matchata.
    """
    total_gt   = 0
    total_tp   = 0   # pred tenuta + match GT
    total_fp   = 0   # pred tenuta + nessun match GT
    total_fn   = 0   # GT non matchata da nessuna pred tenuta

    # Accuracy Qwen su TP reali e FP reali
    qwen_on_real_tp_correct = 0   # Qwen dice yes su TP reale
    qwen_on_real_tp_total   = 0
    qwen_on_real_fp_correct = 0   # Qwen dice no su FP reale
    qwen_on_real_fp_total   = 0

    for r in all_results:
        gt_boxes   = r["gt"]
        pred_boxes = r["preds"][detector]           # [x1,y1,x2,y2] list
        probs      = r["probs"][detector][prompt]   # prob_empty per ogni pred
        is_tp_real = classify_pred_boxes(pred_boxes, gt_boxes, iou_thr)

        n_gt = len(gt_boxes)
        total_gt += n_gt

        # Qwen accuracy su TP/FP reali (indipendente dalla soglia)
        for prob, is_real_tp in zip(probs, is_tp_real):
            if is_real_tp:
                qwen_on_real_tp_total += 1
                if prob >= qwen_thr:   # Qwen dice "è vuoto" → corretto su TP reale
                    qwen_on_real_tp_correct += 1
            else:
                qwen_on_real_fp_total += 1
                if prob < qwen_thr:    # Qwen dice "non è vuoto" → corretto su FP reale
                    qwen_on_real_fp_correct += 1

        # Filtra con soglia Qwen
        kept_boxes = [b for b, p in zip(pred_boxes, probs) if p >= qwen_thr]

        # Match GT con pred tenute
        matched_gt = set()
        for kb in kept_boxes:
            best_iou = 0.0; best_j = -1
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt: continue
                v = iou(kb, gb)
                if v > best_iou:
                    best_iou = v; best_j = j
            if best_iou >= iou_thr and best_j >= 0:
                matched_gt.add(best_j)
                total_tp += 1
            else:
                total_fp += 1

        total_fn += n_gt - len(matched_gt)

    denom_fpr = total_fp + (total_gt - total_tp) if (total_fp + (total_gt - total_tp)) > 0 else 1
    fpr = total_fp / denom_fpr
    fnr = total_fn / total_gt if total_gt > 0 else 0.0
    tpr = total_tp / total_gt if total_gt > 0 else 0.0

    acc_tp = qwen_on_real_tp_correct / qwen_on_real_tp_total if qwen_on_real_tp_total > 0 else None
    acc_fp = qwen_on_real_fp_correct / qwen_on_real_fp_total if qwen_on_real_fp_total > 0 else None

    return {
        "FPR": fpr, "FNR": fnr, "TPR": tpr,
        "TP": total_tp, "FP": total_fp, "FN": total_fn, "GT": total_gt,
        "qwen_acc_on_real_TP": acc_tp,
        "qwen_acc_on_real_FP": acc_fp,
        "qwen_real_TP_total": qwen_on_real_tp_total,
        "qwen_real_FP_total": qwen_on_real_fp_total,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noaug_weights", required=True)
    parser.add_argument("--aug_weights",   required=True)
    parser.add_argument("--qwen_path",     required=True)
    parser.add_argument("--test_images",   required=True)
    parser.add_argument("--test_labels",   required=True)
    parser.add_argument("--out_dir",       required=True)
    parser.add_argument("--conf",          nargs="+", type=float, default=CONF_THRESHOLDS)
    parser.add_argument("--iou_thr",       nargs="+", type=float, default=IOU_THRESHOLDS)
    parser.add_argument("--qwen_thr",      nargs="+", type=float, default=QWEN_THRESHOLDS)
    parser.add_argument("--torch_dtype",   default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max_images",    type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype  = getattr(torch, args.torch_dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Carica detector ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  Valutazione 4 pipeline: 2 detector x 2 prompt Qwen")
    print("="*70)
    print(f"[1/3] Carico detector noaug ← '{args.noaug_weights}'")
    yolo_noaug = YOLO(args.noaug_weights)
    print(f"[2/3] Carico detector aug   ← '{args.aug_weights}'")
    yolo_aug   = YOLO(args.aug_weights)

    # ── Carica Qwen ──────────────────────────────────────────────────────────
    print(f"[3/3] Carico Qwen2.5-VL     ← '{args.qwen_path}'")
    from src.models.qwen_inference_module import QwenInferenceModule
    qwen = QwenInferenceModule(
        model_path  = args.qwen_path,
        torch_dtype = dtype,
        device      = device,
    )

    # ── Carica test set ──────────────────────────────────────────────────────
    pairs = load_test_pairs(Path(args.test_images), Path(args.test_labels))
    if args.max_images:
        pairs = pairs[:args.max_images]
    print(f"\n  Immagini test: {len(pairs)}")
    print(f"  CONF      : {args.conf}")
    print(f"  IOU       : {args.iou_thr}")
    print(f"  QWEN thr  : {args.qwen_thr}")

    # ── Fase 1: YOLO inference ───────────────────────────────────────────────
    print("\nFase 1/2: YOLO inference...")
    detectors = {"noaug": yolo_noaug, "aug": yolo_aug}

    # all_results[conf] = lista di dict per immagine
    all_results: dict[float, list[dict]] = {}

    for conf in args.conf:
        results_conf = []
        for sample in pairs:
            entry = {
                "path": sample["path"],
                "gt":   sample["gt"],
                "image": sample["image"],
                "preds": {},
                "probs": {},
            }
            for det_name, yolo in detectors.items():
                res = yolo(sample["path"], conf=conf, verbose=False)
                if res and res[0].boxes is not None:
                    boxes = res[0].boxes.xyxy.cpu().numpy().tolist()
                else:
                    boxes = []
                entry["preds"][det_name] = boxes
                entry["probs"][det_name] = {}
            results_conf.append(entry)
        all_results[conf] = results_conf
        print(f"  conf={conf}: YOLO completato su {len(results_conf)} immagini")

    # ── Fase 2: Qwen inference ───────────────────────────────────────────────
    print("\nFase 2/2: Qwen inference (immagine intera + BB annotate)...")

    for conf in args.conf:
        for det_name in detectors:
            for prompt_name in PROMPT_NAMES:
                print(f"  det={det_name}  conf={conf}  prompt='{prompt_name}'")
                t0 = time.time()
                for entry in all_results[conf]:
                    boxes = entry["preds"][det_name]
                    if boxes:
                        probs = qwen.score_boxes(
                            image       = entry["image"],
                            boxes       = boxes,
                            prompt_name = prompt_name,
                        )
                    else:
                        probs = []
                    entry["probs"][det_name][prompt_name] = probs
                elapsed = time.time() - t0
                n_boxes = sum(len(e["preds"][det_name]) for e in all_results[conf])
                print(f"    → {n_boxes} box in {elapsed:.1f}s ({elapsed/max(n_boxes,1):.2f}s/box)")

    # ── Fase 3: Metriche ─────────────────────────────────────────────────────
    print("\nCalcolo metriche...")
    output = {}

    for conf in args.conf:
        for iou_t in args.iou_thr:
            for qwen_t in args.qwen_thr:
                for det_name in detectors:
                    for prompt_name in PROMPT_NAMES:
                        key = f"conf={conf}_iou={iou_t}_qwen={qwen_t}_{det_name}_{prompt_name}"
                        m = compute_metrics(
                            all_results[conf], iou_t, qwen_t, det_name, prompt_name
                        )
                        m.update({"conf": conf, "iou": iou_t, "qwen_thr": qwen_t,
                                  "detector": det_name, "prompt": prompt_name})
                        output[key] = m

                        acc_tp_str = f"{m['qwen_acc_on_real_TP']:.3f}" if m['qwen_acc_on_real_TP'] is not None else "N/A"
                        acc_fp_str = f"{m['qwen_acc_on_real_FP']:.3f}" if m['qwen_acc_on_real_FP'] is not None else "N/A"
                        print(
                            f"  {det_name:5s} {prompt_name:12s} "
                            f"conf={conf} iou={iou_t} qwen_thr={qwen_t} | "
                            f"TPR={m['TPR']:.3f} FPR={m['FPR']:.3f} FNR={m['FNR']:.3f} | "
                            f"Qwen acc TP={acc_tp_str} FP={acc_fp_str}"
                        )

    # ── Salva JSON ───────────────────────────────────────────────────────────
    out_json = out_dir / "metrics_v3.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRisultati salvati in: {out_json}")

    # ── Salva riepilogo leggibile ─────────────────────────────────────────────
    out_txt = out_dir / "metrics_v3_summary.txt"
    with open(out_txt, "w") as f:
        f.write("detector | prompt       | conf | iou  | qwen_thr | TPR   | FPR   | FNR   | Qwen_acc_TP | Qwen_acc_FP\n")
        f.write("-" * 110 + "\n")
        for key, m in output.items():
            acc_tp = f"{m['qwen_acc_on_real_TP']:.3f}" if m['qwen_acc_on_real_TP'] is not None else " N/A "
            acc_fp = f"{m['qwen_acc_on_real_FP']:.3f}" if m['qwen_acc_on_real_FP'] is not None else " N/A "
            f.write(
                f"{m['detector']:8s} | {m['prompt']:12s} | {m['conf']:.2f} | "
                f"{m['iou']:.2f} | {m['qwen_thr']:.2f}     | "
                f"{m['TPR']:.3f} | {m['FPR']:.3f} | {m['FNR']:.3f} | "
                f"{acc_tp:11s} | {acc_fp}\n"
            )
    print(f"Riepilogo testuale: {out_txt}")


if __name__ == "__main__":
    main()
