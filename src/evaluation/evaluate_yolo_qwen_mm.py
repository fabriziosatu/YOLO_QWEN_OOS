"""
evaluate_yolo_qwen.py 
-----------------------------
Evaluates 4 pipelines: 2 detectors (noaug / aug) × 2 Qwen prompts (no_context / context)

NEW IN v3:
  - Qwen receives the FULL image with annotated BBs (green = under examination)
  - One query per BB
  - Predicted BBs from YOLO are classified as TP or FP against the Ground Truth
    BEFORE passing them to Qwen, measuring:
      · On real TP BBs: does Qwen say yes (correct) or no (error)?
      · On real FP BBs: does Qwen say yes (error) or no (correct)?
  - Final metrics: FPR, FNR, TPR

Command used to launch evaluate_yolo_qwen_mm.py (Bash):
    
    cd /user/fsaturnino/Progetto_tesi_Qwen_user
    
    nohup env PYTHONPATH=/user/fsaturnino/Progetto_tesi_Qwen_user \
    PYTHONUNBUFFERED=1 python3 -u src/evaluation/evaluate_yolo_qwen_mm.py \
        > /cache/fsaturnino/Progetto_tesi_Qwen/results/baseline_qwen/log.txt 2>&1 &
        
    echo "PID: $!"; tail -f /cache/fsaturnino/Progetto_tesi_Qwen/results/baseline_qwen/log.txt
"""

import argparse, json, time
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════
#  CONFIG — path hardcoded (modificare solo questi)
# ══════════════════════════════════════════════════════════════
WEIGHTS_NOAUG = '/cache/fsaturnino/Progetto_tesi_Qwen/weights/phase1_baseline_comune/no_aug/best.pt'
WEIGHTS_AUG   = '/cache/fsaturnino/Progetto_tesi_Qwen/weights/phase1_baseline_comune/aug/best.pt'
QWEN_PATH     = '/cache/fsaturnino/Progetto_tesi_Qwen/hf_models/Qwen2.5-VL-7B-Instruct'
TEST_IMAGES   = '/cache/fsaturnino/Progetto_tesi_Qwen/dataset_finale/test/images'
TEST_LABELS   = '/cache/fsaturnino/Progetto_tesi_Qwen/dataset_finale/test/labels'
OUT_DIR_DEFAULT = '/cache/fsaturnino/Progetto_tesi_Qwen/results/baseline_qwen'
TORCH_DTYPE   = 'bfloat16'
# ══════════════════════════════════════════════════════════════

CONF_THRESHOLDS  = [0.25, 0.5]
IOU_THRESHOLDS   = [0.0, 0.25, 0.5, 0.75]
QWEN_THRESHOLDS  = [0.3, 0.4, 0.5, 0.6]
PROMPT_NAMES     = ["no_context", "context"]


def load_test_pairs(img_dir: Path, lbl_dir: Path) -> list[dict]:
    """
    Loads and pairs images with their corresponding YOLO format annotations.
    Converts normalized coordinates [center_x, center_y, width, height] into absolute 
    pixel coordinates [x1, y1, x2, y2] based on the original image dimensions.
    """
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
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    Boxes must be formatted as [x1, y1, x2, y2].
    """
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def multi_match(pred_boxes: list, gt_boxes: list, iou_thr: float) -> tuple[list[bool], list[bool]]:
    """
    Performs a multi-matching strategy to evaluate predicted boxes against ground truth.
    A prediction is marked as True Positive (TP) if it overlaps with at least one GT box 
    above the IoU threshold. A GT box is a False Negative (FN) if ignored by all predictions.
    """
    n_preds = len(pred_boxes)
    n_gts   = len(gt_boxes)
    mat_pred = [[] for _ in range(n_preds)]
    mat_gt   = [[] for _ in range(n_gts)]

    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            v = iou(pb, gb)
            if (iou_thr > 0 and v >= iou_thr) or (iou_thr == 0 and v > 0):
                mat_pred[pi].append(gi)
                mat_gt[gi].append(pi)

    is_tp_pred = [len(lst) > 0 for lst in mat_pred]
    is_fn_gt   = [len(lst) == 0 for lst in mat_gt]
    return is_tp_pred, is_fn_gt


def compute_metrics(all_results: list[dict], iou_thr: float, qwen_thr: float,
                    detector: str, prompt: str) -> dict:
    """
    Calculates object detection metrics (FPR, FNR, TPR) and Qwen binary verification accuracy.
    Evaluates how effectively Qwen refines predictions by filtering boxes based on its confidence score,
    and isolates Qwen's specific classification accuracy on real True Positives vs real False Positives.
    """
    total_gt   = 0
    total_tp   = 0
    total_fp   = 0
    total_fn   = 0

    qwen_on_real_tp_correct = 0
    qwen_on_real_tp_total   = 0
    qwen_on_real_fp_correct = 0
    qwen_on_real_fp_total   = 0

    for r in all_results:
        gt_boxes   = r["gt"]
        pred_boxes = r["preds"][detector]
        probs      = r["probs"][detector][prompt]

        n_gt = len(gt_boxes)
        total_gt += n_gt

        is_tp_real, _ = multi_match(pred_boxes, gt_boxes, iou_thr)

        for prob, is_real_tp in zip(probs, is_tp_real):
            if is_real_tp:
                qwen_on_real_tp_total += 1
                if prob >= qwen_thr:
                    qwen_on_real_tp_correct += 1
            else:
                qwen_on_real_fp_total += 1
                if prob < qwen_thr:
                    qwen_on_real_fp_correct += 1

        kept_boxes = [b for b, p in zip(pred_boxes, probs) if p >= qwen_thr]

        if kept_boxes:
            is_tp_kept, is_fn_gt = multi_match(kept_boxes, gt_boxes, iou_thr)
            total_tp += sum(is_tp_kept)
            total_fp += sum(not tp for tp in is_tp_kept)
            total_fn += sum(is_fn_gt)
        else:
            total_fn += n_gt

    fpr = total_fp / total_gt if total_gt > 0 else 0.0
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


def main():
    """
    Main execution pipeline. Handles CLI arguments parsing, model loading (YOLO and Qwen2.5-VL),
    runs sequential batched inference, calculates combined grid-search metrics, and exports
    the structural performance results to JSON and human-readable text summaries.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--noaug_weights", default=WEIGHTS_NOAUG)
    parser.add_argument("--aug_weights",   default=WEIGHTS_AUG)
    parser.add_argument("--qwen_path",     default=QWEN_PATH)
    parser.add_argument("--test_images",   default=TEST_IMAGES)
    parser.add_argument("--test_labels",   default=TEST_LABELS)
    parser.add_argument("--out_dir",       default=OUT_DIR_DEFAULT)
    parser.add_argument("--conf",          nargs="+", type=float, default=CONF_THRESHOLDS)
    parser.add_argument("--iou_thr",       nargs="+", type=float, default=IOU_THRESHOLDS)
    parser.add_argument("--qwen_thr",      nargs="+", type=float, default=QWEN_THRESHOLDS)
    parser.add_argument("--torch_dtype",   default=TORCH_DTYPE, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max_images",    type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype  = getattr(torch, args.torch_dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "="*70)
    print("  Valutazione 4 pipeline: 2 detector x 2 prompt Qwen")
    print("="*70)
    print(f"[1/3] Carico detector noaug ← '{args.noaug_weights}'")
    yolo_noaug = YOLO(args.noaug_weights)
    print(f"[2/3] Carico detector aug   ← '{args.aug_weights}'")
    yolo_aug   = YOLO(args.aug_weights)

    print(f"[3/3] Carico Qwen2.5-VL     ← '{args.qwen_path}'")
    from src.models.qwen_inference_module import QwenInferenceModule
    qwen = QwenInferenceModule(
        model_path  = args.qwen_path,
        torch_dtype = dtype,
        device      = device,
    )

    pairs = load_test_pairs(Path(args.test_images), Path(args.test_labels))
    if args.max_images:
        pairs = pairs[:args.max_images]
    print(f"\n  Immagini test: {len(pairs)}")
    print(f"  CONF      : {args.conf}")
    print(f"  IOU       : {args.iou_thr}")
    print(f"  QWEN thr  : {args.qwen_thr}")

    print("\nFase 1/2: YOLO inference...")
    detectors = {"noaug": yolo_noaug, "aug": yolo_aug}
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

    out_json = out_dir / "metrics_v3.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRisultati salvati in: {out_json}")

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