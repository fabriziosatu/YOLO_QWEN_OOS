"""
evaluate_yolo_qwen_mm_v3.py  —  v3
-----------------------------------
Evaluates 4 pipelines: 2 detectors (noaug / aug) × 2 Qwen prompts (no_context / context)

NEW IN v3 (compared to v2):
  - Qwen responds with three-level text ("yes" / "uncertain" / "no") instead of numerical probabilities
  - qwen_thr is removed: Qwen decides directly with its textual response
  - Input to Qwen: full image with green BB + normalized coordinates in the prompt
  - A close-up crop with 20% padding is passed as a second image to the model for local details
  - UNCERTAIN management logic:
      · "uncertain" cases are counted separately and handled with a secondary forward pass
      · Qwen generates a text explanation of the reason for the ambiguity (saved at cycle completion)
      · In the main metrics of this phase, "uncertain" returns are temporarily treated as "yes"
      · Generates the structured 'uncertain_log.json' log file inside the results directory
  - Metrics: global count of Qwen's YES / NO / UNCERTAIN responses vs YOLO predictions

Usage (Bash):
    cd /user/fsaturnino/Progetto_tesi_Qwen_user
    
    nohup env PYTHONPATH=/user/fsaturnino/Progetto_tesi_Qwen_user \
    PYTHONUNBUFFERED=1 python3 -u src/evaluation/evaluate_yolo_qwen_mm_v3.py \
        > /cache/fsaturnino/Progetto_tesi_Qwen/results/baseline_qwen_v3/log.txt 2>&1 &
        
    echo "PID: $!"; tail -f /cache/fsaturnino/Progetto_tesi_Qwen/results/baseline_qwen_v3/log.txt
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
WEIGHTS_NOAUG   = '/cache/fsaturnino/Progetto_tesi_Qwen/weights/phase1_baseline_comune/no_aug/best.pt'
WEIGHTS_AUG     = '/cache/fsaturnino/Progetto_tesi_Qwen/weights/phase1_baseline_comune/aug/best.pt'
QWEN_PATH       = '/cache/fsaturnino/Progetto_tesi_Qwen/hf_models/Qwen2.5-VL-7B-Instruct'
TEST_IMAGES     = '/cache/fsaturnino/Progetto_tesi_Qwen/dataset_finale/test/images'
TEST_LABELS     = '/cache/fsaturnino/Progetto_tesi_Qwen/dataset_finale/test/labels'
OUT_DIR_DEFAULT = '/cache/fsaturnino/Progetto_tesi_Qwen/results/baseline_qwen_v3'
TORCH_DTYPE     = 'bfloat16'
# ══════════════════════════════════════════════════════════════

CONF_THRESHOLDS = [0.25, 0.5]
IOU_THRESHOLDS  = [0.0, 0.25, 0.5, 0.75]
PROMPT_NAMES    = ["no_context", "context"]


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
        img  = Image.open(img_path).convert("RGB")
        W, H = img.size
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
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def multi_match(pred_boxes: list, gt_boxes: list, iou_thr: float) -> tuple[list[bool], list[bool]]:
    """
    Performs a multi-matching strategy to evaluate predicted boxes against ground truth.
    A prediction is marked as True Positive (TP) if it overlaps with at least one GT box 
    above the IoU threshold. A GT box is a False Negative (FN) if ignored by all predictions.
    """
    mat_pred = [[] for _ in range(len(pred_boxes))]
    mat_gt   = [[] for _ in range(len(gt_boxes))]
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            v = iou(pb, gb)
            if (iou_thr > 0 and v >= iou_thr) or (iou_thr == 0 and v > 0):
                mat_pred[pi].append(gi)
                mat_gt[gi].append(pi)
    is_tp_pred = [len(lst) > 0 for lst in mat_pred]
    is_fn_gt   = [len(lst) == 0 for lst in mat_gt]
    return is_tp_pred, is_fn_gt


def compute_metrics(all_results: list[dict], iou_thr: float, detector: str, prompt: str) -> dict:
    """
    Calculates unified pipeline metrics (FPR, FNR, TPR) based on Qwen's text choices ("yes"/"no").
    Cross-checks verified boxes against Ground Truth, and provides categorical alignment metrics
    quantifying how often Qwen's decisions matched or corrected the standalone YOLO selections.
    """
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    yolo_total       = 0
    qwen_yes_total   = 0
    qwen_no_total    = 0
    qwen_yes_were_tp = 0
    qwen_yes_were_fp = 0
    qwen_no_were_tp  = 0
    qwen_no_were_fp  = 0

    for r in all_results:
        gt_boxes   = r["gt"]
        pred_boxes = r["preds"][detector]
        responses  = r["responses"][detector][prompt]

        n_gt = len(gt_boxes)
        total_gt   += n_gt
        yolo_total += len(pred_boxes)

        is_tp_real, _ = multi_match(pred_boxes, gt_boxes, iou_thr)

        for resp, is_real_tp in zip(responses, is_tp_real):
            if resp == "yes":
                qwen_yes_total += 1
                if is_real_tp:
                    qwen_yes_were_tp += 1
                else:
                    qwen_yes_were_fp += 1
            else:
                qwen_no_total += 1
                if is_real_tp:
                    qwen_no_were_tp += 1
                else:
                    qwen_no_were_fp += 1

        kept_boxes = [b for b, r in zip(pred_boxes, responses) if r == "yes"]

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

    return {
        "FPR": fpr, "FNR": fnr, "TPR": tpr,
        "TP": total_tp, "FP": total_fp, "FN": total_fn, "GT": total_gt,
        "yolo_total":       yolo_total,
        "qwen_yes_total":   qwen_yes_total,
        "qwen_no_total":    qwen_no_total,
        "qwen_yes_were_tp": qwen_yes_were_tp,
        "qwen_yes_were_fp": qwen_yes_were_fp,
        "qwen_no_were_tp":  qwen_no_were_tp,
        "qwen_no_were_fp":  qwen_no_were_fp,
    }


def main():
    """
    Main execution pipeline. Handles CLI arguments parsing, initialization of text-driven VLM 
    with custom uncertainty handling logic and YOLO detectors, processes images to perform 
    validation runs, and exports full logs to text metrics summaries and structural JSON objects.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--noaug_weights", default=WEIGHTS_NOAUG)
    parser.add_argument("--aug_weights",   default=WEIGHTS_AUG)
    parser.add_argument("--qwen_path",     default=QWEN_PATH)
    parser.add_argument("--test_images",   default=TEST_IMAGES)
    parser.add_argument("--test_labels",   default=TEST_LABELS)
    parser.add_argument("--out_dir",       default=OUT_DIR_DEFAULT)
    parser.add_argument("--conf",      nargs="+", type=float, default=CONF_THRESHOLDS)
    parser.add_argument("--iou_thr",  nargs="+", type=float, default=IOU_THRESHOLDS)
    parser.add_argument("--torch_dtype", default=TORCH_DTYPE, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype  = getattr(torch, args.torch_dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 70)
    print("  Valutazione pipeline YOLO + Qwen v5 (yes/uncertain/no + crop 20%)")
    print("=" * 70)
    print(f"[1/3] Carico detector noaug ← '{args.noaug_weights}'")
    yolo_noaug = YOLO(args.noaug_weights)
    print(f"[2/3] Carico detector aug   ← '{args.aug_weights}'")
    yolo_aug   = YOLO(args.aug_weights)

    print(f"[3/3] Carico Qwen2.5-VL     ← '{args.qwen_path}'")
    from src.models.qwen_inference_module_v3 import QwenInferenceModule
    uncertain_log = out_dir / "uncertain_log.json"
    qwen = QwenInferenceModule(
        model_path          = args.qwen_path,
        torch_dtype         = dtype,
        device              = device,
        uncertain_log_path  = str(uncertain_log),
    )

    pairs = load_test_pairs(Path(args.test_images), Path(args.test_labels))
    if args.max_images:
        pairs = pairs[:args.max_images]
    print(f"\n  Immagini test : {len(pairs)}")
    print(f"  CONF          : {args.conf}")
    print(f"  IOU           : {args.iou_thr}")

    print("\nFase 1/2: YOLO inference...")
    detectors   = {"noaug": yolo_noaug, "aug": yolo_aug}
    all_results: dict[float, list[dict]] = {}

    for conf in args.conf:
        results_conf = []
        for sample in pairs:
            entry = {
                "path":      sample["path"],
                "gt":        sample["gt"],
                "image":     sample["image"],
                "preds":     {},
                "responses": {},
            }
            for det_name, yolo in detectors.items():
                res = yolo(sample["path"], conf=conf, verbose=False)
                boxes = res[0].boxes.xyxy.cpu().numpy().tolist() if res and res[0].boxes is not None else []
                entry["preds"][det_name]     = boxes
                entry["responses"][det_name] = {}
            results_conf.append(entry)
        all_results[conf] = results_conf
        print(f"  conf={conf}: YOLO completato su {len(results_conf)} immagini")

    print("\nFase 2/2: Qwen inference (immagine annotata + coordinate nel prompt)...")
    for conf in args.conf:
        for det_name in detectors:
            for prompt_name in PROMPT_NAMES:
                print(f"  det={det_name}  conf={conf}  prompt='{prompt_name}'")
                t0 = time.time()
                yes_count = 0
                no_count  = 0
                uncertain_cycle = 0

                for entry in all_results[conf]:
                    boxes = entry["preds"][det_name]
                    if boxes:
                        resps, unc = qwen.score_boxes(
                            image       = entry["image"],
                            boxes       = boxes,
                            prompt_name = prompt_name,
                            image_name  = str(entry["path"].name),
                        )
                        uncertain_cycle += unc
                    else:
                        resps = []
                    entry["responses"][det_name][prompt_name] = resps
                    yes_count += resps.count("yes")
                    no_count  += resps.count("no")

                elapsed = time.time() - t0
                n_boxes = sum(len(e["preds"][det_name]) for e in all_results[conf])
                print(f"    → {n_boxes} box in {elapsed:.1f}s "
                      f"({elapsed/max(n_boxes,1):.2f}s/box) | "
                      f"Qwen: YES={yes_count}  NO={no_count}  UNCERTAIN={uncertain_cycle}")

    print("\nCalcolo metriche...")
    output = {}

    for conf in args.conf:
        for iou_t in args.iou_thr:
            for det_name in detectors:
                for prompt_name in PROMPT_NAMES:
                    key = f"conf={conf}_iou={iou_t}_{det_name}_{prompt_name}"
                    m   = compute_metrics(
                        all_results[conf], iou_t, det_name, prompt_name
                    )
                    m.update({"conf": conf, "iou": iou_t,
                              "detector": det_name, "prompt": prompt_name})
                    output[key] = m
                    print(
                        f"  {det_name:5s} {prompt_name:12s} "
                        f"conf={conf} iou={iou_t} | "
                        f"TPR={m['TPR']:.3f} FPR={m['FPR']:.3f} FNR={m['FNR']:.3f} | "
                        f"YOLO={m['yolo_total']} "
                        f"Qwen_YES={m['qwen_yes_total']} "
                        f"Qwen_NO={m['qwen_no_total']} | "
                        f"YES_corr={m['qwen_yes_were_tp']} YES_err={m['qwen_yes_were_fp']} "
                        f"NO_corr={m['qwen_no_were_fp']} NO_err={m['qwen_no_were_tp']}"
                    )

    qwen.save_uncertain_log()

    out_json = out_dir / "metrics_v5.json"
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nRisultati salvati in: {out_json}")

    out_txt = out_dir / "metrics_v5_summary.txt"
    with open(out_txt, "w") as f:
        f.write(
            f"{'detector':8s} | {'prompt':12s} | {'conf':4s} | {'iou':4s} | "
            f"{'TPR':5s} | {'FPR':5s} | {'FNR':5s} | "
            f"{'YOLO_tot':8s} | {'Q_YES':5s} | {'Q_NO':5s} | "
            f"{'YES_TP':6s} | {'YES_FP':6s} | {'NO_FP':5s} | {'NO_TP':5s}\n"
        )
        f.write("-" * 130 + "\n")
        for key, m in output.items():
            f.write(
                f"{m['detector']:8s} | {m['prompt']:12s} | {m['conf']:.2f} | "
                f"{m['iou']:.2f} | "
                f"{m['TPR']:.3f} | {m['FPR']:.3f} | {m['FNR']:.3f} | "
                f"{m['yolo_total']:8d} | {m['qwen_yes_total']:5d} | {m['qwen_no_total']:5d} | "
                f"{m['qwen_yes_were_tp']:6d} | {m['qwen_yes_were_fp']:6d} | "
                f"{m['qwen_no_were_fp']:5d} | {m['qwen_no_were_tp']:5d}\n"
            )
    print(f"Riepilogo testuale: {out_txt}")


if __name__ == "__main__":
    main()