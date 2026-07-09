"""
visualize_results.py
====================
Script unificato per visualizzare i risultati di:
  A) YOLO-only (pesi .pt)
  B) YOLO + Qwen (JSON prodotto da evaluate_yolo_qwen.py)

Produce:
  1. Grafico a barre TPR/FPR/FNR (4 subplot per IoU)
  2. Tabella 1 PNG — Metriche principali (Precision/Recall/F1/mAP)
  3. Tabella 2a PNG — FPR e #FP per soglia IoU e conf
  4. Tabella 2c PNG — TPR e #TP per soglia IoU e conf
  5. Tabella statistiche comparative FP/FN per categoria

Uso YOLO-only:
    python3 visualize_results.py --mode yolo \
        --runs \
            detector_only_noaug:/cache/.../best_noaug.pt \
            detector_only_aug:/cache/.../best_aug.pt \
        --images /cache/.../test/images \
        --labels /cache/.../test/labels \
        --split  test \
        --conf   0.25 0.5 \
        --iou    0.0 0.25 0.5 0.75 \
        --out_dir /cache/.../results/vis_yolo

Uso YOLO+Qwen:
    python3 visualize_results.py --mode qwen \
        --json_path /cache/.../results/yolo_qwen_v5/metrics_v3.json \
        --qwen_thr  0.3 \
        --prompt    context \
        --conf      0.25 \
        --iou       0.0 0.25 0.5 0.75 \
        --out_dir   /cache/.../results/vis_qwen
"""

import argparse, json, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
#  FONT
# ─────────────────────────────────────────────────────────────────────────────
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_REG  = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def _font(path, size):
    try:    return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

def _fmt(v, d=3):
    return f"{v:.{d}f}" if v is not None else "—"

def _text_w(text, font):
    try:
        bb = font.getbbox(str(text))
        return bb[2] - bb[0]
    except:
        return len(str(text)) * 7


# ─────────────────────────────────────────────────────────────────────────────
#  COLORI TEMA (ispirato alle tabelle allegate)
# ─────────────────────────────────────────────────────────────────────────────
C_BG       = (255, 252, 248)
C_HDR_GRP  = (180, 140, 100)
C_HDR_SUB  = (210, 180, 150)
C_GRP_BG   = (235, 220, 200)
C_ROW1     = (255, 252, 248)
C_ROW2     = (248, 242, 235)
C_SECTION  = (220, 205, 185)
C_TEXT     = (40,  40,  40)
C_BEST     = (0,  140,   0)
C_LINE     = (200, 185, 170)
C_SEP      = (150, 110,  70)
C_WHITE    = (255, 255, 255)
C_TITLE    = (80,  50,  20)


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY IoU
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + ab - inter + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
#  CARICAMENTO DATI
# ─────────────────────────────────────────────────────────────────────────────

def load_pairs(img_dir: Path, lbl_dir: Path):
    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists(): continue
        from PIL import Image as _I
        img = _I.open(img_path)
        W, H = img.size
        gt = []
        for line in lbl_path.read_text().strip().splitlines():
            p = line.split()
            if len(p) < 5: continue
            _, cx, cy, bw, bh = map(float, p[:5])
            gt.append([(cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H])
        pairs.append({"path": img_path, "gt": gt})
    return pairs


def _multi_match(preds, gts, iou_thr):
    """
    Multi-matching identico a evaluate_yolo_qwen.py (direttive prof).
    Una pred è TP se matcha almeno una GT con IoU sufficiente.
    Una GT è FN se non è matchata da nessuna pred.
    Ritorna (is_tp_pred, matched_gt_indices).
    """
    n_preds = len(preds)
    n_gts   = len(gts)
    mat_pred = [[] for _ in range(n_preds)]
    mat_gt   = [[] for _ in range(n_gts)]

    for pi, pb in enumerate(preds):
        for gi, gb in enumerate(gts):
            v = _iou(pb, gb)
            if (iou_thr > 0 and v >= iou_thr) or (iou_thr == 0 and v > 0):
                mat_pred[pi].append(gi)
                mat_gt[gi].append(pi)

    is_tp_pred = [len(lst) > 0 for lst in mat_pred]
    matched    = {gi for gi, lst in enumerate(mat_gt) if len(lst) > 0}
    return is_tp_pred, matched


def _classify_fp(box):
    x1,y1,x2,y2 = box; bw=x2-x1; bh=y2-y1; area=bw*bh
    if bw<40 or bh<20 or area<800:   return "dark_region"
    if area<1600:                     return "occupied_shelf"
    return "other_fp"

def _classify_fn(box):
    x1,y1,x2,y2 = box; bw=x2-x1; bh=y2-y1; area=bw*bh
    if bw<32 or bh<32 or area<1024:  return "small_box"
    return "other_fn"


def evaluate_yolo_run(yolo, pairs, conf_vals, iou_vals):
    """Valuta un detector YOLO con le stesse formule di evaluate_yolo_qwen.py."""
    from ultralytics import YOLO as _Y
    results = {}
    for conf in conf_vals:
        samples = []
        for s in pairs:
            res   = yolo(s["path"], conf=conf, verbose=False)
            boxes = res[0].boxes.xyxy.cpu().numpy().tolist() \
                    if res and res[0].boxes is not None else []
            samples.append({"gt": s["gt"], "preds": boxes})

        for iou_t in iou_vals:
            total_gt=0; total_tp=0; total_fp=0; total_fn=0
            fp_cats=defaultdict(int); fn_cats=defaultdict(int)
            for s in samples:
                gt=s["gt"]; preds=s["preds"]
                is_tp, matched = _multi_match(preds, gt, iou_t)
                total_gt += len(gt)
                for pb, tp in zip(preds, is_tp):
                    if tp: total_tp+=1
                    else:  total_fp+=1; fp_cats[_classify_fp(pb)]+=1
                for j,gb in enumerate(gt):
                    if j not in matched:
                        total_fn+=1; fn_cats[_classify_fn(gb)]+=1

            results[(conf,iou_t)] = {
                "GT":total_gt,"TP":total_tp,"FP":total_fp,"FN":total_fn,
                "TPR": total_tp/total_gt if total_gt>0 else 0.0,
                "FPR": total_fp/total_gt if total_gt>0 else 0.0,
                "FNR": total_fn/total_gt if total_gt>0 else 0.0,
                "fp_cats": dict(fp_cats),
                "fn_cats": dict(fn_cats),
            }

    # Ultralytics per Tabella 1
    try:
        uv = yolo.val(verbose=False)
        rd = uv.results_dict
        p  = float(rd.get("metrics/precision(B)",0))
        r  = float(rd.get("metrics/recall(B)",   0))
        results["ultralytics"] = {
            "precision": p, "recall": r,
            "f1":        2*p*r/(p+r) if (p+r)>0 else 0.0,
            "map50":     float(rd.get("metrics/mAP50(B)",    0)),
            "map50_95":  float(rd.get("metrics/mAP50-95(B)", 0)),
        }
    except Exception as e:
        print(f"  [WARN] Ultralytics val: {e}")
        results["ultralytics"] = {}

    return results


def load_qwen_results(json_path, conf_vals, iou_vals, qwen_thrs, prompt):
    """Carica risultati per lista di qwen_thr. FPR = FP/GT per coerenza con YOLO-only."""
    with open(json_path) as f:
        data = json.load(f)
    run_names = sorted({v["detector"] for v in data.values()})
    results_by_thr = {}
    for qwen_thr in qwen_thrs:
        all_results = {}
        for det in run_names:
            res = {}
            for conf in conf_vals:
                for iou_t in iou_vals:
                    k = f"conf={conf}_iou={iou_t}_qwen={qwen_thr}_{det}_{prompt}"
                    if k not in data: continue
                    m = data[k]
                    res[(conf, iou_t)] = {
                        "GT":  m["GT"],  "TP": m["TP"],
                        "FP":  m["FP"],  "FN": m["FN"],
                        "TPR": m["TPR"],
                        "FPR": m["FP"] / m["GT"] if m["GT"] > 0 else 0.0,
                        "FNR": m["FNR"],
                        "fp_cats": m.get("fp_cats", {}),
                        "fn_cats": m.get("fn_cats", {}),
                        "qwen_acc_tp": m.get("qwen_acc_on_real_TP"),
                        "qwen_acc_fp": m.get("qwen_acc_on_real_FP"),
                    }
            res["ultralytics"] = {}
            all_results[det] = res
        results_by_thr[qwen_thr] = all_results
    return run_names, results_by_thr


# ─────────────────────────────────────────────────────────────────────────────
#  1. GRAFICO A BARRE
# ─────────────────────────────────────────────────────────────────────────────

def plot_bars(all_results, run_names, conf_val, iou_vals, title, out_path):
    fig, axes = plt.subplots(1, len(iou_vals), figsize=(6*len(iou_vals), 6),
                             sharey=False)
    if len(iou_vals) == 1: axes = [axes]

    bar_colors = {"TPR": "#2ecc71", "FPR": "#e74c3c", "FNR": "#f39c12"}
    w = 0.22

    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, iou_t in zip(axes, iou_vals):
        x = np.arange(len(run_names))
        for mi, (metric, mc) in enumerate(bar_colors.items()):
            vals = [all_results[rn].get((conf_val, iou_t), {}).get(metric, 0)
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
        ax.set_xticklabels([n.replace("_"," ").title() for n in run_names], fontsize=9)
        ax.set_ylabel("Valore metrica"); ax.set_ylim(0, 1.18)
        ax.grid(axis="y", alpha=0.3)
        if iou_t == iou_vals[0]:
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Grafico barre → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  2. TABELLA PNG — engine
# ─────────────────────────────────────────────────────────────────────────────

def _make_png_table(title_text, cols, col_widths, all_rows, get_val_fn,
                    best_cols, lower_is_better_cols, int_cols,
                    footer_text, out_path, conf_thresholds):
    """
    Genera una tabella PNG con header a due livelli per colonne raggruppate
    (colonne con '/c' nel nome vengono raggruppate).
    """
    ROW_H  = 30; GRP_H = 26
    HDR_R1 = 20; HDR_R2 = 20; HDR_H = HDR_R1 + HDR_R2
    PAD    = 14; TITLE_H = 26

    n_data = sum(1 for r in all_rows if r[0] != "__group__")
    n_grp  = sum(1 for r in all_rows if r[0] == "__group__")
    total_h = TITLE_H + HDR_H + n_data*ROW_H + n_grp*GRP_H + 44
    total_w = sum(col_widths) + PAD*2 + 4

    F_T   = _font(FONT_BOLD, 13); F_H1 = _font(FONT_BOLD, 11)
    F_H2  = _font(FONT_REG,  10); F_C  = _font(FONT_REG,  11)
    F_M   = _font(FONT_BOLD, 11)

    img  = Image.new("RGB", (total_w, total_h), C_BG)
    draw = ImageDraw.Draw(img)

    def x_col(ci): return PAD + sum(col_widths[:ci])
    def is_grp(col): return "/c" in col

    def draw_centered(txt, x, y, w, h, font, color):
        tw = _text_w(txt, font)
        draw.text((x+(w-tw)//2, y+(h-13)//2), str(txt), fill=color, font=font)

    def hline(y, width=1, color=None):
        draw.line([(PAD,y),(total_w-PAD,y)], fill=color or C_LINE, width=width)

    def vline(x, y1, y2, width=1, color=None):
        draw.line([(x,y1),(x,y2)], fill=color or C_SEP, width=width)

    # Titolo
    draw.text((PAD, 4), title_text, fill=C_TITLE, font=F_T)

    # Header
    y_hdr = TITLE_H
    draw.rectangle([PAD, y_hdr, total_w-PAD, y_hdr+HDR_H], fill=C_HDR_SUB)
    n_conf = len(conf_thresholds)
    grp_ci = 0
    for ci, (col, w) in enumerate(zip(cols, col_widths)):
        x = x_col(ci)
        if not is_grp(col):
            draw.rectangle([x, y_hdr, x+w, y_hdr+HDR_H], fill=C_HDR_GRP)
            ty = y_hdr+(HDR_H-13)//2
            tx = x+6 if ci<=1 else x+(w-_text_w(col,F_H1))//2
            draw.text((tx,ty), col, fill=C_WHITE, font=F_H1)
        else:
            ci_in = grp_ci % n_conf
            if ci_in == 0:
                gw = sum(col_widths[ci:ci+n_conf])
                draw.rectangle([x,y_hdr,x+gw,y_hdr+HDR_R1], fill=C_HDR_GRP)
                lbl = col.split("/c")[0].replace("_"," ")
                tw = _text_w(lbl, F_H1)
                draw.text((x+(gw-tw)//2, y_hdr+(HDR_R1-13)//2), lbl, fill=C_WHITE, font=F_H1)
                vline(x, y_hdr, y_hdr+HDR_H)
            sub_lbl = "c"+col.split("/c")[1]
            draw.rectangle([x, y_hdr+HDR_R1, x+w, y_hdr+HDR_H], fill=C_HDR_SUB)
            draw.text((x+(w-_text_w(sub_lbl,F_H2))//2, y_hdr+HDR_R1+(HDR_R2-11)//2),
                      sub_lbl, fill=C_WHITE, font=F_H2)
            draw.line([(x,y_hdr+HDR_R1),(x+w,y_hdr+HDR_R1)], fill=C_SEP, width=1)
            grp_ci += 1

    hline(y_hdr+HDR_H, width=2)
    y = y_hdr+HDR_H

    # Calcola best (solo righe dati)
    data_rows = [(l,s) for l,s in all_rows if l != "__group__"]
    def best_val(col):
        vals = [get_val_fn(l,s,col) for l,s in data_rows]
        vals = [v for v in vals if v is not None]
        if not vals: return None
        return min(vals) if col in lower_is_better_cols else max(vals)
    bests = {col: best_val(col) for col in best_cols}

    # Righe
    di = 0
    for label, split in all_rows:
        if label == "__group__":
            draw.rectangle([PAD,y,total_w-PAD,y+GRP_H], fill=C_GRP_BG)
            draw.text((PAD+8, y+(GRP_H-13)//2), split, fill=C_TITLE, font=F_M)
            hline(y+GRP_H, width=2); y+=GRP_H; continue

        bg = C_ROW1 if di%2==0 else C_ROW2
        draw.rectangle([PAD,y,total_w-PAD,y+ROW_H], fill=bg)
        draw.text((x_col(0)+6, y+(ROW_H-13)//2), label, fill=C_TEXT, font=F_M)
        draw.text((x_col(1)+6, y+(ROW_H-13)//2), split, fill=C_TEXT, font=F_C)

        grp_c2 = 0
        for ci, col in enumerate(cols[2:], start=2):
            val = get_val_fn(label, split, col)
            if col in int_cols and val is not None:
                txt = str(int(val))
            else:
                txt = _fmt(val) if val is not None else "—"
            best  = bests.get(col)
            isbest = (val is not None and best is not None and abs(val-best)<1e-6)
            draw_centered(txt, x_col(ci), y, col_widths[ci], ROW_H,
                          F_C, C_BEST if isbest else C_TEXT)
            if is_grp(col):
                if grp_c2 % n_conf == 0:
                    vline(x_col(ci), y, y+ROW_H, color=C_LINE)
                grp_c2 += 1

        hline(y+ROW_H); y+=ROW_H; di+=1

    draw.rectangle([PAD,TITLE_H,total_w-PAD,y+4], outline=C_LINE, width=1)
    draw.text((PAD, y+8), footer_text, fill=(120,100,80), font=F_C)
    img.save(out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  3. TABELLA 1 — Metriche principali
# ─────────────────────────────────────────────────────────────────────────────

def make_table1(all_results, run_names, split_label, out_path, mode):
    cols  = ["Run","Split","Precision","Recall","F1","mAP@50","mAP@50-95",
             "empty","box_loss","cls_loss","dfl_loss","tot_loss"]
    cw    = [160,55,82,68,62,76,96,66,82,82,82,82]

    all_rows = [("__group__", split_label)]
    for rn in run_names:
        all_rows.append((rn, split_label))

    def get_val(label, split, col):
        u = all_results[label].get("ultralytics", {})
        return {"Precision": u.get("precision"),
                "Recall":    u.get("recall"),
                "F1":        u.get("f1"),
                "mAP@50":    u.get("map50"),
                "mAP@50-95": u.get("map50_95"),
                "empty":     u.get("map50"),
                "box_loss":  None, "cls_loss":  None,
                "dfl_loss":  None, "tot_loss":  None,
               }.get(col)

    best_cols = ["Precision","Recall","F1","mAP@50","mAP@50-95","empty"]
    note = ("Verde=miglior valore per colonna  |  Precision=P@conf0.25 (Ultralytics default)  "
            "|  empty=mAP@50 per classe  |  Loss: val ultima epoca")
    if mode == "qwen":
        note = ("Verde=miglior valore per colonna  |  "
                "Tabella 1 non disponibile per YOLO+Qwen — usare Tabella 2a/2c per confronto diretto")

    _make_png_table(
        title_text="Tabella 1 — Metriche principali",
        cols=cols, col_widths=cw, all_rows=all_rows,
        get_val_fn=get_val, best_cols=best_cols,
        lower_is_better_cols=["box_loss","cls_loss","dfl_loss","tot_loss"],
        int_cols=set(), footer_text=note, out_path=out_path,
        conf_thresholds=[0.25],
    )
    print(f"  [OK] Tabella 1 → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  4. TABELLE 2a (FPR) e 2c (TPR)
# ─────────────────────────────────────────────────────────────────────────────

def make_table2(all_results, run_names, conf_vals, iou_vals,
                split_label, out_dir, conf_thresholds):
    fpr_cols = [f"FPR_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    tpr_cols = [f"TPR_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    fp_cols  = [f"#FP_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    tp_cols  = [f"#TP_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    W = 78

    all_rows = [("__group__", split_label)]
    for rn in run_names:
        all_rows.append((rn, split_label))

    def get_val(label, split, col):
        for iou_t in iou_vals:
            for conf_t in conf_vals:
                m = all_results[label].get((conf_t, iou_t), {})
                if col == f"FPR_IoU@{iou_t}/c{conf_t}": return m.get("FPR")
                if col == f"TPR_IoU@{iou_t}/c{conf_t}": return m.get("TPR")
                if col == f"#FP_IoU@{iou_t}/c{conf_t}": return m.get("FP")
                if col == f"#TP_IoU@{iou_t}/c{conf_t}": return m.get("TP")
                if col == "#GT": return m.get("GT")
        return None

    # 2a FPR
    _make_png_table(
        title_text=f"Tabella 2a — FPR (#pred con lista vuota / #GT)\nRaggruppata per IoU  |  conf: {conf_vals}",
        cols=["Run","Split","#GT",*fpr_cols,*fp_cols],
        col_widths=[160,55,52,*[W]*len(fpr_cols),*[W]*len(fp_cols)],
        all_rows=all_rows, get_val_fn=get_val,
        best_cols=fpr_cols, lower_is_better_cols=fpr_cols,
        int_cols={*fp_cols,"#GT"},
        footer_text=f"Verde=valore più basso  |  FPR=FP/GT  |  conf: {conf_vals}  |  IoU: {iou_vals} (Multi-match)",
        out_path=str(out_dir/"tabella2a_fpr.png"),
        conf_thresholds=conf_thresholds,
    )
    print(f"  [OK] Tabella 2a → {out_dir/'tabella2a_fpr.png'}")

    # 2c TPR
    _make_png_table(
        title_text=f"Tabella 2c — TPR (#match pred↔GT / #GT)\nRaggruppata per IoU  |  conf: {conf_vals}",
        cols=["Run","Split","#GT",*tpr_cols,*tp_cols],
        col_widths=[160,55,52,*[W]*len(tpr_cols),*[W]*len(tp_cols)],
        all_rows=all_rows, get_val_fn=get_val,
        best_cols=tpr_cols, lower_is_better_cols=[],
        int_cols={*tp_cols,"#GT"},
        footer_text=f"Verde=valore più alto  |  TPR=TP/GT  |  conf: {conf_vals}  |  IoU: {iou_vals} (Multi-match)",
        out_path=str(out_dir/"tabella2c_tpr.png"),
        conf_thresholds=conf_thresholds,
    )
    print(f"  [OK] Tabella 2c → {out_dir/'tabella2c_tpr.png'}")


# ─────────────────────────────────────────────────────────────────────────────
#  5. TABELLA STATISTICHE COMPARATIVE FP/FN
# ─────────────────────────────────────────────────────────────────────────────

def make_stat_table(all_results, run_names, conf_val, split_label, out_path):
    fp_cats = ["dark_region","motion_blur","occupied_shelf","refrigerator","other_fp"]
    fn_cats = ["small_box","dark_fn","low_contrast","occluded","other_fn"]
    fp_desc = {"dark_region":"Zona nera / bordo scaffale",
               "motion_blur":"Blur da movimento camera",
               "occupied_shelf":"Zona con prodotti",
               "refrigerator":"Cella frigo",
               "other_fp":"Altro FP"}
    fn_desc = {"small_box":"GT box piccola (<32px)",
               "dark_fn":"Zona d'ombra",
               "low_contrast":"Basso contrasto",
               "occluded":"Occlusione GT",
               "other_fn":"Altro FN"}

    # Larghezze colonne
    col_run_w = 170
    col_cat_w = 160
    col_val_w = 140
    col_des_w = 220
    n_runs    = len(run_names)
    total_w   = col_cat_w + n_runs*col_val_w + col_des_w + 32
    ROW_H=30; SEC_H=28; HDR_H=52; TITLE_H=26
    total_h = TITLE_H + HDR_H + (len(fp_cats)+len(fn_cats)+2)*ROW_H + 2*SEC_H + 44

    F_T = _font(FONT_BOLD,13); F_H = _font(FONT_BOLD,11)
    F_C = _font(FONT_REG, 11); F_M = _font(FONT_BOLD,11)

    img  = Image.new("RGB",(total_w,total_h),C_BG)
    draw = ImageDraw.Draw(img)
    PAD  = 14

    def hline(y,width=1,color=None):
        draw.line([(PAD,y),(total_w-PAD,y)],fill=color or C_LINE,width=width)

    def col_x(ci):
        xs = [PAD, PAD+col_cat_w]
        for i in range(n_runs): xs.append(xs[-1]+col_val_w)
        xs.append(xs[-1]+col_des_w)
        return xs[ci]

    # Titolo
    draw.text((PAD,4),
              f"Statistiche comparative — Analisi FP/FN  (conf={conf_val}, IoU=0.0)",
              fill=C_TITLE, font=F_T)

    # Intestazione colonne
    y = TITLE_H
    draw.rectangle([PAD,y,total_w-PAD,y+HDR_H], fill=C_HDR_GRP)
    draw.text((col_x(0)+6, y+(HDR_H-13)//2), "Categoria",   fill=C_WHITE, font=F_H)
    draw.text((col_x(-1)+6,y+(HDR_H-13)//2), "Descrizione", fill=C_WHITE, font=F_H)

    # Intestazione run con riepilogo numerico
    for ri, rn in enumerate(run_names):
        m    = all_results[rn].get((conf_val, 0.0), {})
        gt   = m.get("GT", 0); fp = m.get("FP",0)
        fn   = m.get("FN",0);  fnr= m.get("FNR",0)
        hdr  = f"{rn}"
        sub  = f"GT={gt} FP={fp} FN={fn} FNR={fnr:.1%}"
        cx   = col_x(ri+1)
        tw1  = _text_w(hdr,F_H)
        draw.text((cx+(col_val_w-tw1)//2, y+4),          hdr, fill=C_WHITE, font=F_H)
        tw2  = _text_w(sub,F_C)
        draw.text((cx+(col_val_w-tw2)//2, y+HDR_H-16),   sub, fill=(240,230,220), font=F_C)
    hline(y+HDR_H, width=2)
    y += HDR_H

    def draw_section(label):
        nonlocal y
        draw.rectangle([PAD,y,total_w-PAD,y+SEC_H], fill=C_SECTION)
        draw.text((col_x(0)+8, y+(SEC_H-13)//2), label, fill=C_TITLE, font=F_M)
        hline(y+SEC_H, width=1); y+=SEC_H

    def draw_row(cat, cat_dict, desc_dict, di):
        nonlocal y
        bg = C_ROW1 if di%2==0 else C_ROW2
        draw.rectangle([PAD,y,total_w-PAD,y+ROW_H], fill=bg)
        draw.text((col_x(0)+6, y+(ROW_H-13)//2), cat, fill=C_TEXT, font=F_C)
        draw.text((col_x(-1)+6,y+(ROW_H-13)//2), desc_dict.get(cat,cat), fill=C_TEXT, font=F_C)

        vals = []
        for ri, rn in enumerate(run_names):
            m   = all_results[rn].get((0.25, 0.0), {})
            tot = m.get(cat_dict+"_tot", 1) or 1
            n   = m.get(cat_dict+"_cats",{}).get(cat,0)
            vals.append((n, tot))

        best_n = max(v[0] for v in vals) if vals else 0
        for ri, (n, tot) in enumerate(vals):
            pct = n/tot*100
            txt = f"{n} ({pct:.0f}%)"
            cx  = col_x(ri+1)
            tw  = _text_w(txt, F_C)
            color = C_BEST if n==best_n and best_n>0 else C_TEXT
            draw.text((cx+(col_val_w-tw)//2, y+(ROW_H-13)//2), txt, fill=color, font=F_C)

        hline(y+ROW_H); y+=ROW_H

    # Recupera fp_cats e fn_cats con chiavi corrette
    for rn in run_names:
        m = all_results[rn].get((conf_val, 0.0), {})
        m["fp_cats_tot"] = m.get("FP",1) or 1
        m["fn_cats_tot"] = m.get("FN",1) or 1

    draw_section("FALSI POSITIVI (FP)")
    for di, cat in enumerate(fp_cats):
        # Patch: inietta i valori nel formato atteso da draw_row
        for rn in run_names:
            m = all_results[rn].get((conf_val,0.0),{})
            m["fp_cats_cats"] = m.get("fp_cats",{})
            m["fp_cats_tot"]  = m.get("FP",1) or 1
        draw_row(cat, "fp_cats", fp_desc, di)

    draw_section("FALSI NEGATIVI (FN)")
    for di, cat in enumerate(fn_cats):
        for rn in run_names:
            m = all_results[rn].get((conf_val,0.0),{})
            m["fn_cats_cats"] = m.get("fn_cats",{})
            m["fn_cats_tot"]  = m.get("FN",1) or 1
        draw_row(cat, "fn_cats", fn_desc, di)

    draw.rectangle([PAD,TITLE_H,total_w-PAD,y+4], outline=C_LINE, width=1)
    draw.text((PAD,y+8),
              "Verde=valore più alto per riga  |  Soglie: dark<40  blur<50  small<32px",
              fill=(120,100,80), font=F_C)
    img.save(out_path)
    print(f"  [OK] Tabella stat → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      required=True, choices=["yolo","qwen"])
    # YOLO
    parser.add_argument("--runs",      nargs="+",
                        help="nome:path_pesi  (solo --mode yolo)")
    parser.add_argument("--images",    help="cartella immagini (solo --mode yolo)")
    parser.add_argument("--labels",    help="cartella label   (solo --mode yolo)")
    parser.add_argument("--split",     default="test")
    # Qwen
    parser.add_argument("--json_path", help="JSON di evaluate_yolo_qwen.py (solo --mode qwen)")
    parser.add_argument("--qwen_thr",  nargs="+", type=float, default=[0.3])
    parser.add_argument("--prompt",    default="context",
                        choices=["context","no_context"])
    # Comuni
    parser.add_argument("--conf",      nargs="+", type=float, default=[0.25,0.5])
    parser.add_argument("--iou",       nargs="+", type=float, default=[0.0,0.25,0.5,0.75])
    parser.add_argument("--bar_conf",  type=float, default=0.25)
    parser.add_argument("--out_dir",   required=True)
    parser.add_argument("--max_images",type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Carica dati ───────────────────────────────────────────────────────────
    if args.mode == "yolo":
        from ultralytics import YOLO as _Y
        run_names = []; run_paths = {}
        for r in args.runs:
            name, path = r.split(":",1)
            run_names.append(name); run_paths[name] = path

        pairs = load_pairs(Path(args.images), Path(args.labels))
        if args.max_images: pairs = pairs[:args.max_images]
        print(f"Immagini: {len(pairs)}  GT totali: {sum(len(s['gt']) for s in pairs)}")

        all_results = {}
        for name in run_names:
            print(f"\n[{name}] Valuto...")
            yolo = _Y(run_paths[name])
            all_results[name] = evaluate_yolo_run(yolo, pairs, args.conf, args.iou)

        split_label = args.split.upper()
        bar_title   = (f"Pipeline completa (YOLO) — conf={args.bar_conf}  split={args.split}\n"
                       f"{len(run_names)} run × {len(args.iou)} soglie IoU")

    else:  # qwen
        run_names, results_by_thr = load_qwen_results(
            args.json_path, args.conf, args.iou, args.qwen_thr, args.prompt)
        split_label = "TEST"
        for qwen_thr in args.qwen_thr:
            thr_str     = str(qwen_thr).replace(".", "")
            thr_out_dir = out_dir / f"qwen_thr_{thr_str}"
            thr_out_dir.mkdir(parents=True, exist_ok=True)
            all_results = results_by_thr[qwen_thr]
            bar_title   = (f"Pipeline YOLO + Qwen — conf={args.bar_conf}  "
                           f"qwen_thr={qwen_thr}  prompt={args.prompt}\n"
                           f"{len(run_names)} detector x {len(args.iou)} soglie IoU")
            print(f"\n{'='*50}\n  qwen_thr={qwen_thr}  output: {thr_out_dir}\n{'='*50}")
            print("  [1/4] Grafico a barre...")
            plot_bars(all_results, run_names, args.bar_conf, args.iou,
                      bar_title, thr_out_dir/"bar_metrics.png")
            print("  [2/4] Tabella 1...")
            make_table1(all_results, run_names, split_label,
                        thr_out_dir/"tabella1_metriche.png", args.mode)
            print("  [3/4] Tabelle 2a/2c...")
            make_table2(all_results, run_names, args.conf, args.iou,
                        split_label, thr_out_dir, args.conf)
            print("  [4/4] Statistiche FP/FN...")
            make_stat_table(all_results, run_names, args.bar_conf,
                            split_label, thr_out_dir/"statistiche_comparative.png")
        print(f"\nTutto salvato in: {out_dir}")
        return

    # ── Output YOLO-only ──────────────────────────────────────────────────────
    print(f"\n[1/4] Grafico a barre...")
    plot_bars(all_results, run_names, args.bar_conf, args.iou,
              bar_title, out_dir/"bar_metrics.png")
    print(f"\n[2/4] Tabella 1 (metriche principali)...")
    make_table1(all_results, run_names, split_label,
                out_dir/"tabella1_metriche.png", args.mode)
    print(f"\n[3/4] Tabelle 2a/2c (FPR/TPR)...")
    make_table2(all_results, run_names, args.conf, args.iou,
                split_label, out_dir, args.conf)
    print(f"\n[4/4] Tabella statistiche FP/FN...")
    make_stat_table(all_results, run_names, args.bar_conf,
                    split_label, out_dir/"statistiche_comparative.png")
    print(f"\nTutto salvato in: {out_dir}")


if __name__ == "__main__":
    main()