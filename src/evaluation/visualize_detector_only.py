"""
visualize_detector_only.py  —  v1.0
------------------------------------
Evaluation and grid-search summary visualization suite for standalone YOLO detectors (noaug and aug).

KEY FEATURES:
  - Runs evaluation passes on the common test set to measure the baseline performance of the detectors.
  - Implements a multi-match strategy to evaluate predicted boxes against Ground Truth parameters.
  - Automatically generates and exports metric tables directly into PNG image formats:
      · Table 2a — False Positive Rate (FPR) and total number of FPs.
      · Table 2b — False Negative Rate (FNR) and total number of FNs.
      · Table 2c — True Positive Rate (TPR) and total number of TPs.
  - Produces multi-panel bar charts grouped by different Intersection over Union (IoU) thresholds.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════
#  CONFIG — Hardcoded paths
# ══════════════════════════════════════════════════════════════
WEIGHTS_NOAUG = '/cache/fsaturnino/Progetto_tesi_Qwen/weights/phase1_baseline_comune/no_aug/best.pt'
WEIGHTS_AUG   = '/cache/fsaturnino/Progetto_tesi_Qwen/weights/phase1_baseline_comune/aug/best.pt'

IMG_DIR   = '/cache/fsaturnino/Progetto_tesi_Qwen/dataset_finale/test/images'
LBL_DIR   = '/cache/fsaturnino/Progetto_tesi_Qwen/dataset_finale/test/labels'
OUTPUT_DIR = '/user/fsaturnino/Progetto_tesi_Qwen_user/BASELINE'

CONF_THRESHOLDS = [0.25, 0.5]
IOU_THRESHOLDS  = [0.0, 0.25, 0.5, 0.75]
SPLIT_LABEL     = 'TEST'

# ══════════════════════════════════════════════════════════════
#  FONT INITIALIZATION
# ══════════════════════════════════════════════════════════════
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_REG  = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def _font(path, size):
    try:    return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

# ══════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════

def load_test_pairs(img_dir, lbl_dir):
    """
    Loads text labels and pairing images, converting center-relative 
    coordinates [center_x, center_y, width, height] into absolute pixel bounds.
    """
    pairs = []
    for img_path in sorted(Path(img_dir).glob('*.jpg')) + \
                    sorted(Path(img_dir).glob('*.png')):
        lbl_path = Path(lbl_dir) / (img_path.stem + '.txt')
        if not lbl_path.exists():
            continue
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        gt_boxes = []
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw/2) * W; y1 = (cy - bh/2) * H
            x2 = (cx + bw/2) * W; y2 = (cy + bh/2) * H
            gt_boxes.append([x1, y1, x2, y2])
        pairs.append({'path': img_path, 'gt': gt_boxes})
    return pairs


def iou(a, b):
    """Computes the Intersection over Union (IoU) metric between two absolute bounding boxes."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    if inter == 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)


def multi_match(preds, gts, iou_thr):
    """
    Executes multi-matching evaluation strategy. 
    A candidate prediction acts as a True Positive if it overlaps with at least one 
    Ground Truth box above the specified IoU threshold.
    """
    mp = [[] for _ in preds]; mg = [[] for _ in gts]
    for pi, pb in enumerate(preds):
        for gi, gb in enumerate(gts):
            v = iou(pb, gb)
            if (iou_thr > 0 and v >= iou_thr) or (iou_thr == 0 and v > 0):
                mp[pi].append(gi); mg[gi].append(pi)
    is_tp = [len(x) > 0 for x in mp]
    is_fn = [len(x) == 0 for x in mg]
    return is_tp, is_fn


def evaluate(yolo, pairs, conf_vals, iou_vals):
    """
    Sweeps through combinations of confidence thresholds and IoU limits to compile a metric profile 
    dictionary containing aggregated statistical outputs (GT, TP, FP, FN, TPR, FPR, FNR).
    """
    results = {}
    for conf in conf_vals:
        preds_all = []
        for s in pairs:
            res = yolo(s['path'], conf=conf, verbose=False)
            boxes = res[0].boxes.xyxy.cpu().numpy().tolist() \
                    if res and res[0].boxes is not None else []
            preds_all.append(boxes)

        for iou_t in iou_vals:
            total_gt = total_tp = total_fp = total_fn = 0
            for s, preds in zip(pairs, preds_all):
                gts = s['gt']
                total_gt += len(gts)
                if preds and gts:
                    is_tp, is_fn = multi_match(preds, gts, iou_t)
                    total_tp += sum(is_tp)
                    total_fp += sum(not t for t in is_tp)
                    total_fn += sum(is_fn)
                elif preds and not gts:
                    total_fp += len(preds)
                elif not preds and gts:
                    total_fn += len(gts)

            gt = total_gt or 1
            results[(conf, iou_t)] = {
                'GT': total_gt, 'TP': total_tp,
                'FP': total_fp, 'FN': total_fn,
                'TPR': total_tp / gt,
                'FPR': total_fp / gt,
                'FNR': total_fn / gt,
            }
    return results


# ══════════════════════════════════════════════════════════════
#  BAR CHART RENDERING
# ══════════════════════════════════════════════════════════════

def plot_bars(run_results, run_names, conf_val, iou_vals, title, out_path):
    """Generates multi-pane comparative graphical charts for TPR/FPR/FNR metrics grouped by IoU step."""
    fig, axes = plt.subplots(1, len(iou_vals), figsize=(6*len(iou_vals), 6), sharey=False)
    if len(iou_vals) == 1:
        axes = [axes]

    bar_colors = {"TPR": "#2ecc71", "FPR": "#e74c3c", "FNR": "#f39c12"}
    w = 0.22

    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, iou_t in zip(axes, iou_vals):
        x = np.arange(len(run_names))
        for mi, (metric, mc) in enumerate(bar_colors.items()):
            vals = [run_results[rn].get((conf_val, iou_t), {}).get(metric, 0)
                    for rn in run_names]
            offset = (mi - 1) * w
            rects  = ax.bar(x + offset, vals, w, color=mc, alpha=0.88,
                            edgecolor="white", linewidth=0.5, label=metric)
            for r, v in zip(rects, vals):
                if v > 0.02:
                    ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.01,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"IoU={iou_t}", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_", " ").title() for n in run_names], fontsize=9)
        ax.set_ylabel("Metric Value")
        ax.set_ylim(0, 1.18)
        ax.grid(axis="y", alpha=0.3)
        if iou_t == iou_vals[0]:
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════
#  PNG TABLE RASTERIZATION ENGINE
# ══════════════════════════════════════════════════════════════

def _text_w(text, font):
    try:    return font.getlength(text)
    except: return len(text) * 7


def _make_png_table(title_text, cols, col_widths, all_rows,
                    get_val_fn, best_cols, lower_is_better_cols,
                    int_cols, footer_text, out_path, conf_thresholds):
    """Renders structured summary metric matrices directly into publication-quality PNG table graphics."""
    F_TITLE  = _font(FONT_BOLD, 13)
    F_HDR    = _font(FONT_BOLD, 10)
    F_SUB    = _font(FONT_BOLD,  9)
    F_CELL   = _font(FONT_REG,  10)
    F_FOOT   = _font(FONT_REG,   9)

    C_BG      = (255, 252, 248)
    C_HDR_BG  = (139, 109,  71)
    C_HDR_FG  = (255, 255, 255)
    C_GRP_BG  = (210, 190, 160)
    C_ROW1    = (255, 252, 248)
    C_ROW2    = (245, 240, 230)
    C_TEXT    = (40,  40,  40)
    C_GREEN   = (0,  140,  0)
    C_LINE    = (200, 185, 170)
    C_TITLE   = (80,  50,  20)

    TITLE_H = 28; HDR_H = 22; SUB_H = 18; GRP_H = 20; ROW_H = 22; FOOT_H = 24; PAD = 10

    total_w = PAD*2 + sum(col_widths)
    data_rows = [r for r in all_rows if r[0] != '__group__']
    groups    = [(i, r[1]) for i, r in enumerate(all_rows) if r[0] == '__group__']

    # Compute column local optima values to emphasize best results
    best_vals = {}
    for col in cols[3:]:
        vals = []
        for label, split in data_rows:
            v = get_val_fn(label, split, col)
            if v is not None:
                try: vals.append(float(v))
                except: pass
        if vals:
            best_vals[col] = min(vals) if col in lower_is_better_cols else max(vals)

    n_data = len(data_rows)
    n_grp  = len(groups)
    total_h = TITLE_H + HDR_H + SUB_H + n_grp*GRP_H + n_data*ROW_H + FOOT_H + 8

    img  = Image.new('RGB', (total_w, total_h), C_BG)
    draw = ImageDraw.Draw(img)

    def col_x(ci):
        x = PAD
        for i in range(ci): x += col_widths[i]
        return x

    def hline(y, color=None, width=1):
        draw.line([(PAD, y), (total_w-PAD, y)], fill=color or C_LINE, width=width)

    def cell(text, ci, y, h, font, color, bg=None, align='center'):
        x = col_x(ci); w = col_widths[ci]
        if bg: draw.rectangle([x, y, x+w, y+h], fill=bg)
        tw = _text_w(str(text), font)
        tx = x + (w-tw)//2 if align == 'center' else x+4
        draw.text((tx, y+(h-11)//2), str(text), fill=color, font=font)

    # Render main title block
    draw.text((PAD, 6), title_text, fill=C_TITLE, font=F_TITLE)

    # Render primary tier header
    y = TITLE_H
    draw.rectangle([PAD, y, total_w-PAD, y+HDR_H], fill=C_HDR_BG)
    cell('Run',   0, y, HDR_H, F_HDR, C_HDR_FG, align='left')
    cell('Split', 1, y, HDR_H, F_HDR, C_HDR_FG)
    cell('#GT',   2, y, HDR_H, F_HDR, C_HDR_FG)
    ci = 3
    for col in cols[3:]:
        cell(col, ci, y, HDR_H, F_HDR, C_HDR_FG)
        ci += 1
    hline(y+HDR_H, color=(100,80,50), width=2)
    y += HDR_H

    # Render secondary tier configuration sub-headers (c0.25 / c0.5)
    draw.rectangle([PAD, y, total_w-PAD, y+SUB_H], fill=C_HDR_BG)
    ci = 3
    for col in cols[3:]:
        for k, cv in enumerate(conf_thresholds):
            sub_w = col_widths[ci] // len(conf_thresholds)
            sx = col_x(ci) + k * sub_w
            tw = _text_w(f'c{cv}', F_SUB)
            draw.text((sx+(sub_w-tw)//2, y+(SUB_H-9)//2), f'c{cv}',
                      fill=C_HDR_FG, font=F_SUB)
        ci += 1
    hline(y+SUB_H, color=(100,80,50), width=1)
    y += SUB_H

    # Righe dati con gruppi
    grp_ptr = 0
    di = 0
    for label, split in data_rows:
        # Riga gruppo se necessario
        while grp_ptr < len(groups) and groups[grp_ptr][0] == di + grp_ptr:
            draw.rectangle([PAD, y, total_w-PAD, y+GRP_H], fill=C_GRP_BG)
            draw.text((PAD+6, y+(GRP_H-11)//2), groups[grp_ptr][1], fill=C_TEXT, font=F_HDR)
            hline(y+GRP_H)
            y += GRP_H
            grp_ptr += 1

        row_bg = C_ROW1 if di % 2 == 0 else C_ROW2
        draw.rectangle([PAD, y, total_w-PAD, y+ROW_H], fill=row_bg)

        cell(label, 0, y, ROW_H, F_CELL, C_TEXT, align='left')
        cell(split, 1, y, ROW_H, F_CELL, C_TEXT)

        gt_val = get_val_fn(label, split, '#GT')
        cell(str(gt_val) if gt_val is not None else '—', 2, y, ROW_H, F_CELL, C_TEXT)

        ci = 3
        for col in cols[3:]:
            v = get_val_fn(label, split, col)
            if v is None:
                cell('—', ci, y, ROW_H, F_CELL, C_TEXT)
            else:
                is_best = col in best_vals and abs(float(v) - best_vals[col]) < 1e-9
                color   = C_GREEN if is_best else C_TEXT
                txt     = str(int(v)) if col in int_cols else f'{float(v):.3f}'
                cell(txt, ci, y, ROW_H, F_CELL, color)
            ci += 1

        hline(y+ROW_H)
        y += ROW_H
        di += 1

    # Finalize frame bounding structures
    draw.rectangle([PAD, y, total_w-PAD, y+FOOT_H], fill=C_BG)
    draw.text((PAD, y+6), footer_text, fill=(120,100,80), font=F_FOOT)
    draw.rectangle([PAD, TITLE_H, total_w-PAD, y+FOOT_H-2], outline=C_LINE, width=1)

    img.save(out_path)
    print(f'  Saved: {out_path}')


def make_tables(run_results, out_dir, conf_vals, iou_vals, split_label):
    """Produce tabella2a (FPR), tabella2b (FNR), tabella2c (TPR)."""
    os.makedirs(out_dir, exist_ok=True)
    run_names = list(run_results.keys())

    iou_labels   = [f'IoU@{t}' for t in iou_vals]
    n_iou = len(iou_vals)

    cw_run  = 180; cw_split = 55; cw_gt = 45; cw_iou = 110; cw_cnt = 85
    all_rows = [('__group__', split_label)] + [(rn, split_label) for rn in run_names]

    for metric, metric_label, count_key, count_label, title, footer, best_low in [
        ('FPR', 'FPR', 'FP', '#FP',
         'Tabella 2a — FPR (#pred on empty locations / #GT)',
         f'Green=lowest metric value  |  FPR=FP/GT  |  conf: {conf_vals}  |  IoU: {iou_vals}  (Multi-match)',
         True),
        ('FNR', 'FNR', 'FN', '#FN',
         'Tabella 2b — FNR (#GT components left empty / #GT)',
         f'Green=lowest metric value  |  FNR=FN/GT  |  conf: {conf_vals}  |  IoU: {iou_vals}  (Multi-match)',
         True),
        ('TPR', 'TPR', 'TP', '#TP',
         'Tabella 2c — TPR (#matched pred-GT pairs / #GT)',
         f'Green=highest metric value  |  TPR=TP/GT  |  conf: {conf_vals}  |  IoU: {iou_vals}  (Multi-match)',
         False),
    ]:
        cols2 = ['Run', 'Split', '#GT']
        for iou_lbl in iou_labels: cols2.append(f'{metric_label} {iou_lbl}')
        for iou_lbl in iou_labels: cols2.append(f'{count_label} {iou_lbl}')

        col_widths = [cw_run, cw_split, cw_gt] + [cw_iou]*n_iou + [cw_cnt]*n_iou
        int_cols = {f'{count_label} {iou_lbl}' for iou_lbl in iou_labels} | {'#GT'}

        def make_get_val(metric, count_key, iou_labels, iou_vals, conf_vals, metric_label, count_label):
            def get_val(label, split, col):
                if label not in run_results: return None
                if col == '#GT':
                    r = run_results[label].get((conf_vals[0], iou_vals[0]))
                    return r['GT'] if r else None
                for iou_lbl, iou_t in zip(iou_labels, iou_vals):
                    if col == f'{metric_label} {iou_lbl}':
                        r = run_results[label].get((conf_vals[0], iou_t))
                        return r[metric] if r else None
                    if col == f'{count_label} {iou_lbl}':
                        r = run_results[label].get((conf_vals[0], iou_t))
                        return r[count_key] if r else None
                return None
            return get_val

        gv = make_get_val(metric, count_key, iou_labels, iou_vals, conf_vals, metric_label, count_label)
        out_name = {'FPR': 'tabella2a_fpr.png', 'FNR': 'tabella2b_fnr.png', 'TPR': 'tabella2c_tpr.png'}[metric]

        _make_png_table(
            title_text=title, cols=cols2, col_widths=col_widths, all_rows=all_rows, get_val_fn=gv,
            best_cols=[c for c in cols2[3:]] if best_low else [],
            lower_is_better_cols=[c for c in cols2[3:]] if best_low else [],
            int_cols=int_cols, footer_text=footer, out_path=os.path.join(out_dir, out_name),
            conf_thresholds=conf_vals,
        )


# ══════════════════════════════════════════════════════════════
#  MAIN SYSTEM ENTRYPOINT
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 60)
    print('  Valutazione Detector Only — TEST SET')
    print('=' * 60)

    print(f'\n[1/2] Loading noaug  ← {WEIGHTS_NOAUG}')
    yolo_noaug = YOLO(WEIGHTS_NOAUG)
    print(f'[2/2] Loading aug    ← {WEIGHTS_AUG}')
    yolo_aug   = YOLO(WEIGHTS_AUG)

    pairs = load_test_pairs(IMG_DIR, LBL_DIR)
    print(f'\nTest images parsed: {len(pairs)}')
    gt_total = sum(len(s['gt']) for s in pairs)
    print(f'Total Ground Truth markers: {gt_total}')

    print('\nEvaluating noaug pipeline...')
    res_noaug = evaluate(yolo_noaug, pairs, CONF_THRESHOLDS, IOU_THRESHOLDS)
    print('Evaluating aug pipeline...')
    res_aug   = evaluate(yolo_aug,   pairs, CONF_THRESHOLDS, IOU_THRESHOLDS)

    print('\nRiepilogo (conf=0.25, IoU=0.0):')
    for name, res in [('noaug', res_noaug), ('aug', res_aug)]:
        m = res[(0.25, 0.0)]
        print(f'  {name:8s}  TPR={m["TPR"]:.3f}  FPR={m["FPR"]:.3f}  '
              f'FNR={m["FNR"]:.3f}  GT={m["GT"]}  '
              f'TP={m["TP"]}  FP={m["FP"]}  FN={m["FN"]}')

    print('\nGenerazione grafici e tabelle PNG...')
    run_results = {
        'detector_only_noaug': res_noaug,
        'detector_only_aug':   res_aug,
    }
    run_names = list(run_results.keys())

    bar_title = (f"YOLO-only — conf={CONF_THRESHOLDS[0]}  split=TEST\n"
                 f"{len(run_names)} detector x {len(IOU_THRESHOLDS)} soglie IoU")
    plot_bars(run_results, run_names, CONF_THRESHOLDS[0], IOU_THRESHOLDS,
              bar_title, os.path.join(OUTPUT_DIR, "bar_metrics.png"))

    make_tables(run_results, OUTPUT_DIR, CONF_THRESHOLDS, IOU_THRESHOLDS, SPLIT_LABEL)

    print(f'\nOutput: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()