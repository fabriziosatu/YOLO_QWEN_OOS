"""
src/evaluation/analyze_errors.py
==================================
Analisi automatica degli errori FP/FN — supporto multi-modello.

Confronta N modelli sullo stesso test set in una sola run, producendo:
  - CSV e crop per ogni modello separatamente
  - pannelli visivi ricchi (immagine intera + cerchio + crop ingrandita)
  - comparison.png      : grafici a barre side-by-side
  - comparison.txt      : tabella testo
  - error_table.png     : tabella riassuntiva stile paper (PIL)
  - delta.png           : differenze % tra due modelli

Uso (singolo modello):
    python src\evaluation\analyze_errors.py ^
        --weights weights\phase1_final\weights\best.pt ^
        --data_dir data\dataset_finale --split test

Uso (confronto due modelli):
    python src\evaluation\analyze_errors.py ^
        --weights weights\cluster_training\phase1_yolo_final\weights\best.pt ^
                  weights\cluster_training\phase1_yolo_final_aug\weights\best.pt ^
        --names phase1_final phase1_aug ^
        --data_dir data\dataset_finale --split test

Categorie FP: dark_region, motion_blur, refrigerator, occupied_shelf, other_fp
Categorie FN: small_box, dark_fn, occluded, low_contrast, other_fn
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO


# ── Soglie calibrabili ────────────────────────────────────────────────────────

THR_DARK_BRIGHTNESS   = 40
THR_BLUR_LAPLACIAN    = 100
THR_REFRIG_BLUE_FRAC  = 0.38
THR_REFRIG_BRIGHTNESS = 120
THR_REFRIG_SAT_MAX    = 40      # [NUOVO] saturazione BASSA per celle frigo
THR_OCCUPIED_SAT      = 40
THR_OCCUPIED_VAR      = 400.0
THR_SMALL_BOX_PX      = 32
THR_LOW_CONTRAST_VAR  = 200.0
THR_OCCLUDED_IOU      = 0.3

IMG_SIZE          = 640
MAX_CROPS_PER_CAT = 30          # crop PNG semplici per categoria
N_PANNELLI        = 8           # pannelli ricchi per categoria

FP_CATS = ["dark_region", "motion_blur", "refrigerator", "occupied_shelf", "other_fp"]
FN_CATS = ["small_box", "dark_fn", "occluded", "low_contrast", "other_fn"]

DESC_FP = {
    "dark_region":    "Zona nera / bordo scaffale",
    "motion_blur":    "Blur da movimento camera",
    "refrigerator":   "Cella frigo",
    "occupied_shelf": "Zona con prodotti",
    "other_fp":       "Altro FP",
}
DESC_FN = {
    "small_box":    "GT box piccola (<32px)",
    "dark_fn":      "Zona d'ombra",
    "occluded":     "Occlusione GT",
    "low_contrast": "Basso contrasto",
    "other_fn":     "Altro FN",
}

CAT_COLORS = {
    "dark_region":    (30,  30, 200),
    "motion_blur":    (200,200,  30),
    "refrigerator":   (30, 180, 200),
    "occupied_shelf": (200, 80,  30),
    "other_fp":       (150,150, 150),
    "small_box":      (200, 30,  30),
    "dark_fn":        (30,  30, 150),
    "occluded":       (200,130,  30),
    "low_contrast":   (150, 30, 200),
    "other_fn":       (150,150, 150),
}


# ── Utility IoU ───────────────────────────────────────────────────────────────

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)


def iou_matrix(preds, gts):
    if len(preds) == 0 or len(gts) == 0:
        return np.zeros((len(preds), len(gts)))
    mat = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            mat[i, j] = iou_xyxy(p, g)
    return mat


def match_predictions(pred_boxes, gt_boxes, iou_thr):
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return [], [], []
    if len(pred_boxes) == 0:
        return [], [], list(range(len(gt_boxes)))
    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []

    order       = np.argsort(-pred_boxes[:, 4])
    pred_sorted = pred_boxes[order]
    iou_mat     = iou_matrix(pred_sorted[:, :4], gt_boxes)
    matched_gt  = set()
    tp_s, fp_s  = [], []

    for i in range(len(pred_sorted)):
        best_gt = int(np.argmax(iou_mat[i]))
        if iou_mat[i, best_gt] >= iou_thr and best_gt not in matched_gt:
            tp_s.append(i); matched_gt.add(best_gt)
        else:
            fp_s.append(i)

    fn_idx = [j for j in range(len(gt_boxes)) if j not in matched_gt]
    return [int(order[i]) for i in tp_s], [int(order[i]) for i in fp_s], fn_idx


# ── Euristiche visive ─────────────────────────────────────────────────────────

def crop_image(img_bgr, box_xyxy, margin=0.05):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = np.array(box_xyxy[:4]).astype(int)
    dx = int((x2 - x1) * margin); dy = int((y2 - y1) * margin)
    x1 = max(0, x1 - dx); y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx); y2 = min(h, y2 + dy)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_bgr[y1:y2, x1:x2]


def compute_features(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return dict(brightness=0., laplacian_var=0., saturation=0.,
                    pixel_var=0., blue_frac=0., area_px=0)
    hsv           = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    brightness    = float(hsv[:, :, 2].mean())
    gray          = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturation    = float(hsv[:, :, 1].mean())
    pixel_var     = float(gray.var())
    total_int     = crop_bgr.astype(float).sum(axis=2).mean()
    blue_frac     = float(crop_bgr[:, :, 0].mean()) / (total_int / 3 + 1e-6)
    return dict(brightness=brightness, laplacian_var=laplacian_var,
                saturation=saturation, pixel_var=pixel_var,
                blue_frac=blue_frac, area_px=crop_bgr.shape[0]*crop_bgr.shape[1])


def classify_fp(f):
    if f["brightness"]    < THR_DARK_BRIGHTNESS:
        return "dark_region"
    if f["laplacian_var"] < THR_BLUR_LAPLACIAN:
        return "motion_blur"
    # [MIGLIORATO] aggiunto vincolo saturation < THR_REFRIG_SAT_MAX
    # per escludere zone colorate che non sono celle frigo
    if (f["blue_frac"]   > THR_REFRIG_BLUE_FRAC
            and f["brightness"] > THR_REFRIG_BRIGHTNESS
            and f["saturation"] < THR_REFRIG_SAT_MAX):
        return "refrigerator"
    if (f["saturation"]  > THR_OCCUPIED_SAT
            and f["pixel_var"] > THR_OCCUPIED_VAR):
        return "occupied_shelf"
    return "other_fp"


def classify_fn(f, box, all_gt, idx):
    w = box[2] - box[0]; h = box[3] - box[1]
    if min(w, h) < THR_SMALL_BOX_PX:
        return "small_box"
    if f["brightness"] < THR_DARK_BRIGHTNESS:
        return "dark_fn"
    if len(all_gt) > 1:
        for j, other in enumerate(all_gt):
            if j != idx and iou_xyxy(box, other) > THR_OCCLUDED_IOU:
                return "occluded"
    if f["pixel_var"] < THR_LOW_CONTRAST_VAR:
        return "low_contrast"
    return "other_fn"


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_gt_boxes(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return np.zeros((0, 4))
    with open(label_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    if int(parts[0]) != 0:
                        continue
                    xc, yc, bw, bh = map(float, parts[1:])
                    boxes.append([(xc-bw/2)*img_w, (yc-bh/2)*img_h,
                                  (xc+bw/2)*img_w, (yc+bh/2)*img_h])
                except ValueError:
                    continue
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4))


def save_simple_crop(crop_bgr, out_path, size=96):
    if crop_bgr is None or crop_bgr.size == 0:
        return
    cv2.imwrite(str(out_path), cv2.resize(crop_bgr, (size, size)))


# ── Pannelli visivi ricchi ────────────────────────────────────────────────────

def _pil_font(size=10):
    for name in ["DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"]:
        for base in ["/usr/share/fonts/truetype/dejavu/",
                     "/usr/share/fonts/truetype/liberation/",
                     "C:/Windows/Fonts/"]:
            p = Path(base) / name
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except Exception:
                    pass
    return ImageFont.load_default()


def save_rich_panel(img_bgr, box_xyxy, feats, cat, extra_txt, out_path):
    """
    Salva un pannello con:
      - colonna sinistra : immagine intera scalata + rettangolo + cerchio
      - colonna destra   : crop ingrandita con bordo colorato
      - riga info        : categoria, features numeriche
    """
    IMG_MAX  = 480
    CROP_MAX = 300
    INFO_H   = 56
    HDR_H    = 20
    PAD      = 6
    BG       = (25, 25, 25)
    C_WHITE  = (240, 240, 240)
    C_GRAY   = (160, 160, 160)

    color = CAT_COLORS.get(cat, (180, 180, 180))
    h_orig, w_orig = img_bgr.shape[:2]

    # Immagine intera scalata
    scale = min(IMG_MAX / max(w_orig, 1), IMG_MAX / max(h_orig, 1), 1.0)
    dw = max(1, int(w_orig * scale)); dh = max(1, int(h_orig * scale))
    img_pil  = Image.fromarray(cv2.cvtColor(
        cv2.resize(img_bgr, (dw, dh)), cv2.COLOR_BGR2RGB))
    draw_img = ImageDraw.Draw(img_pil)

    # Box scalata
    bx1 = int(box_xyxy[0] * scale); by1 = int(box_xyxy[1] * scale)
    bx2 = int(box_xyxy[2] * scale); by2 = int(box_xyxy[3] * scale)
    bcx = (bx1 + bx2) // 2;        bcy = (by1 + by2) // 2
    draw_img.rectangle([bx1, by1, bx2, by2], outline=color, width=2)
    # Cerchio attorno alla box
    r = int(((max(bx2-bx1,1)**2 + max(by2-by1,1)**2)**0.5) / 2) + 14
    draw_img.ellipse([bcx-r, bcy-r, bcx+r, bcy+r], outline=color, width=3)

    # Crop ingrandita
    crop_bgr = crop_image(img_bgr, np.array(box_xyxy))
    if crop_bgr is None or crop_bgr.size == 0:
        crop_bgr = img_bgr
    ch, cw = crop_bgr.shape[:2]
    sc = min(CROP_MAX / max(cw, 1), CROP_MAX / max(ch, 1), 4.0)
    nw = max(1, int(cw * sc)); nh = max(1, int(ch * sc))
    crop_pil  = Image.fromarray(cv2.cvtColor(
        cv2.resize(crop_bgr, (nw, nh)), cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(crop_pil).rectangle([0,0,nw-1,nh-1], outline=color, width=3)

    # Canvas
    cw_tot = PAD + dw + PAD + nw + PAD
    ch_tot = HDR_H + max(dh, nh) + PAD + INFO_H
    canvas = Image.new("RGB", (cw_tot, ch_tot), BG)
    draw   = ImageDraw.Draw(canvas)
    f_sm   = _pil_font(9); f_bold = _pil_font(10)

    draw.rectangle([PAD, 0, PAD+dw, HDR_H], fill=(50,50,60))
    draw.text((PAD+4, 4), "Immagine intera", fill=C_WHITE, font=f_sm)
    xc = PAD+dw+PAD
    draw.rectangle([xc, 0, xc+nw, HDR_H], fill=(50,50,60))
    draw.text((xc+4, 4), f"Crop x{sc:.1f}", fill=C_WHITE, font=f_sm)

    canvas.paste(img_pil,  (PAD, HDR_H))
    canvas.paste(crop_pil, (xc,  HDR_H))

    ty = HDR_H + max(dh, nh) + 4
    draw.text((PAD, ty),    cat,      fill=color,  font=f_bold)
    draw.text((PAD, ty+16),
              f"bright={feats['brightness']:.0f}  lap={feats['laplacian_var']:.0f}  "
              f"sat={feats['saturation']:.0f}  var={feats['pixel_var']:.0f}",
              fill=C_GRAY, font=f_sm)
    draw.text((PAD, ty+30), extra_txt, fill=C_GRAY, font=f_sm)

    canvas.save(str(out_path))


# ── Analisi singolo modello ───────────────────────────────────────────────────

def analyze_one_model(model, model_name, img_paths, lbl_dir,
                      out_path, iou_thr, conf_thr):
    import random
    model_out = out_path / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    # Cartelle crop semplici
    for cat in FP_CATS:
        (model_out / "crops" / "fp" / cat).mkdir(parents=True, exist_ok=True)
    for cat in FN_CATS:
        (model_out / "crops" / "fn" / cat).mkdir(parents=True, exist_ok=True)
    # Cartelle pannelli ricchi
    for cat in FP_CATS:
        (model_out / "pannelli" / "fp" / cat).mkdir(parents=True, exist_ok=True)
    for cat in FN_CATS:
        (model_out / "pannelli" / "fn" / cat).mkdir(parents=True, exist_ok=True)

    crop_cnt_fp = {c: 0 for c in FP_CATS}
    crop_cnt_fn = {c: 0 for c in FN_CATS}
    pan_cnt_fp  = {c: 0 for c in FP_CATS}
    pan_cnt_fn  = {c: 0 for c in FN_CATS}
    fp_counts   = {c: 0 for c in FP_CATS}
    fn_counts   = {c: 0 for c in FN_CATS}
    n_gt_total  = 0
    n_images    = 0

    fp_fields = ["image","x1","y1","x2","y2","conf",
                 "brightness","laplacian_var","saturation","pixel_var",
                 "blue_frac","area_px","categoria"]
    fn_fields = ["image","x1","y1","x2","y2",
                 "brightness","laplacian_var","saturation","pixel_var",
                 "blue_frac","area_px","categoria"]

    with open(model_out/"fp_errors.csv","w",newline="",encoding="utf-8") as fp_f, \
         open(model_out/"fn_errors.csv","w",newline="",encoding="utf-8") as fn_f:

        fp_w = csv.DictWriter(fp_f, fieldnames=fp_fields); fp_w.writeheader()
        fn_w = csv.DictWriter(fn_f, fieldnames=fn_fields); fn_w.writeheader()

        for img_path in tqdm(img_paths, desc=f"  [{model_name}]", ncols=90):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_bgr  = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
            h, w     = img_bgr.shape[:2]
            n_images += 1

            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if not lbl_path.exists():
                lbl_path = img_path.with_suffix(".txt")
            gt_boxes = load_gt_boxes(lbl_path, w, h)
            n_gt_total += len(gt_boxes)

            results    = model.predict(str(img_path), conf=conf_thr,
                                       verbose=False, imgsz=IMG_SIZE)
            pred_xyxy  = results[0].boxes.xyxy.cpu().numpy()
            pred_conf  = results[0].boxes.conf.cpu().numpy()
            pred_boxes = (np.concatenate([pred_xyxy, pred_conf[:,None]], axis=1)
                          if len(pred_xyxy) > 0 else np.zeros((0,5)))

            _, fp_idx, fn_idx = match_predictions(pred_boxes, gt_boxes, iou_thr)

            # ── FP ───────────────────────────────────────────────────────────
            for i in fp_idx:
                box   = pred_boxes[i,:4]
                crop  = crop_image(img_bgr, box)
                feats = compute_features(crop)
                cat   = classify_fp(feats)
                fp_counts[cat] += 1
                fp_w.writerow({"image": img_path.name,
                               "x1":round(float(box[0]),1),"y1":round(float(box[1]),1),
                               "x2":round(float(box[2]),1),"y2":round(float(box[3]),1),
                               "conf":round(float(pred_boxes[i,4]),3),
                               **{k:round(v,3) for k,v in feats.items()},
                               "categoria":cat})
                if crop_cnt_fp[cat] < MAX_CROPS_PER_CAT and crop is not None:
                    save_simple_crop(crop,
                        model_out/"crops"/"fp"/cat/f"{img_path.stem}_fp{i}.png")
                    crop_cnt_fp[cat] += 1
                if pan_cnt_fp[cat] < N_PANNELLI:
                    save_rich_panel(img_bgr, box, feats, cat,
                        f"conf={pred_boxes[i,4]:.3f}  "
                        f"size={int(box[2]-box[0])}x{int(box[3]-box[1])}px",
                        model_out/"pannelli"/"fp"/cat/
                            f"{img_path.stem}_fp{i}_{cat}.png")
                    pan_cnt_fp[cat] += 1

            # ── FN ───────────────────────────────────────────────────────────
            for j in fn_idx:
                box   = gt_boxes[j]
                crop  = crop_image(img_bgr, box)
                feats = compute_features(crop)
                cat   = classify_fn(feats, box, gt_boxes, j)
                fn_counts[cat] += 1
                bw = int(box[2]-box[0]); bh = int(box[3]-box[1])
                fn_w.writerow({"image": img_path.name,
                               "x1":round(float(box[0]),1),"y1":round(float(box[1]),1),
                               "x2":round(float(box[2]),1),"y2":round(float(box[3]),1),
                               **{k:round(v,3) for k,v in feats.items()},
                               "categoria":cat})
                if crop_cnt_fn[cat] < MAX_CROPS_PER_CAT and crop is not None:
                    save_simple_crop(crop,
                        model_out/"crops"/"fn"/cat/f"{img_path.stem}_fn{j}.png")
                    crop_cnt_fn[cat] += 1
                if pan_cnt_fn[cat] < N_PANNELLI:
                    save_rich_panel(img_bgr, box, feats, cat,
                        f"size={bw}x{bh}px  area={bw*bh}px2",
                        model_out/"pannelli"/"fn"/cat/
                            f"{img_path.stem}_fn{j}_{cat}.png")
                    pan_cnt_fn[cat] += 1

    return fp_counts, fn_counts, n_gt_total, n_images


# ── Tabella PNG comparativa (PIL) ─────────────────────────────────────────────

def make_error_table_png(all_results, out_path):
    """
    Tabella riassuntiva stile paper con PIL.
    Verde = valore piu alto per riga.
    """
    n = len(all_results)
    ROW_H  = 28; HDR_H = 38; GRP_H = 32; PAD = 14
    CAT_W  = 175; DESC_W = 200; RUN_W = 130
    total_w = PAD + CAT_W + n * RUN_W + DESC_W + PAD
    n_rows  = 1 + len(FP_CATS) + 1 + len(FN_CATS)
    total_h = HDR_H + n_rows * ROW_H + 2 * GRP_H + 52

    C_BG    = (255,252,248); C_HDR  = (210,180,150); C_GRP  = (240,225,210)
    C_R1    = (255,252,248); C_R2   = (248,242,235); C_TEXT = (40,40,40)
    C_LINE  = (200,185,170); C_WH   = (255,255,255); C_BEST = (0,130,0)

    img  = Image.new("RGB", (total_w, total_h), C_BG)
    draw = ImageDraw.Draw(img)
    F    = _pil_font(10); FB = _pil_font(11)

    def x_col(i):
        return PAD + CAT_W + i * RUN_W if i < n else PAD + CAT_W + n * RUN_W

    def cell(txt, x, y, w, h, font=None, color=C_TEXT, bg=None, align="left"):
        font = font or F
        if bg:
            draw.rectangle([x, y, x+w, y+h], fill=bg)
        try:
            bb = font.getbbox(str(txt)); tw = bb[2]-bb[0]
        except Exception:
            tw = len(str(txt)) * 7
        tx = x+6 if align == "left" else x+(w-tw)//2
        draw.text((tx, y+(h-12)//2), str(txt), fill=color, font=font)

    def hline(y_):
        draw.line([(PAD, y_), (total_w-PAD, y_)], fill=C_LINE, width=1)

    # ── Header ───────────────────────────────────────────────────────────────
    y = 0
    cell("Categoria", PAD, y, CAT_W, HDR_H, FB, C_WH, C_HDR)
    for i, r in enumerate(all_results):
        fp_tot = sum(r["fp_counts"].values())
        fn_tot = sum(r["fn_counts"].values())
        fnr    = 100*fn_tot/max(r["n_gt_total"],1)
        info   = f"GT={r['n_gt_total']} FP={fp_tot} FN={fn_tot} FNR={fnr:.1f}%"
        xi     = x_col(i)
        draw.rectangle([xi, y, xi+RUN_W, HDR_H], fill=C_HDR)
        draw.text((xi + RUN_W//2 - 30, y+4),  r["name"], fill=C_WH, font=FB)
        draw.text((xi+4, y+20), info, fill=(230,220,200), font=F)
    cell("Descrizione", x_col(n), y, DESC_W, HDR_H, FB, C_WH, C_HDR)
    hline(HDR_H); y += HDR_H

    def section(title, cats, counts_key, totals_key, descs):
        nonlocal y
        draw.rectangle([PAD, y, total_w-PAD, y+GRP_H], fill=C_GRP)
        draw.text((PAD+8, y+(GRP_H-12)//2), title, fill=C_TEXT, font=FB)
        hline(y+GRP_H); y += GRP_H
        max_per_cat = {
            c: max(r[counts_key][c] for r in all_results) for c in cats
        }
        for ri, cat in enumerate(cats):
            bg = C_R1 if ri%2==0 else C_R2
            draw.rectangle([PAD, y, total_w-PAD, y+ROW_H], fill=bg)
            cell(cat, PAD, y, CAT_W, ROW_H)
            for i, r in enumerate(all_results):
                cnt = r[counts_key][cat]
                tot = sum(r[totals_key].values()) if isinstance(r[totals_key], dict) \
                      else r[totals_key]
                pct = 100*cnt/tot if tot > 0 else 0
                col = C_BEST if (cnt == max_per_cat[cat] and cnt > 0) else C_TEXT
                cell(f"{cnt} ({pct:.0f}%)", x_col(i), y, RUN_W, ROW_H,
                     color=col, align="center")
            cell(descs.get(cat,""), x_col(n), y, DESC_W, ROW_H, color=(100,80,60))
            hline(y+ROW_H); y += ROW_H

    section("FALSI POSITIVI (FP)", FP_CATS, "fp_counts", "fp_counts", DESC_FP)
    section("FALSI NEGATIVI (FN)", FN_CATS, "fn_counts", "fn_counts", DESC_FN)

    # Footer soglie
    draw.text((PAD, y+4),
        f"Verde=valore piu alto per riga  |  "
        f"Soglie: dark<{THR_DARK_BRIGHTNESS}  blur<{THR_BLUR_LAPLACIAN}  "
        f"small<{THR_SMALL_BOX_PX}px  sat>{THR_OCCUPIED_SAT}  "
        f"tex>{THR_OCCUPIED_VAR}  occ>{THR_OCCLUDED_IOU}  "
        f"refrig>{THR_REFRIG_BLUE_FRAC}",
        fill=(130,110,90), font=F)
    draw.rectangle([PAD, 0, total_w-PAD, y+46], outline=C_LINE, width=1)

    out_file = out_path / "error_table.png"
    img.save(str(out_file))
    print(f"  -> Tabella PNG salvata: {out_file}")


# ── Grafici matplotlib + delta ────────────────────────────────────────────────

def make_comparison(all_results, out_path, n_images):
    lines = ["="*70,
             f"  CONFRONTO MODELLI -- {n_images} immagini, split test",
             "="*70]
    for r in all_results:
        fp_tot = sum(r["fp_counts"].values())
        fn_tot = sum(r["fn_counts"].values())
        fnr    = 100*fn_tot/max(r["n_gt_total"],1)
        fppi   = fp_tot / max(r["n_images"],1)
        lines.append(f"\n  -- {r['name']} --")
        lines.append(f"  GT:{r['n_gt_total']}  FP:{fp_tot}  FN:{fn_tot}  "
                     f"FNR:{fnr:.1f}%  FPPI:{fppi:.3f}")
        for cat in FP_CATS:
            cnt = r["fp_counts"][cat]
            pct = 100*cnt/fp_tot if fp_tot>0 else 0
            lines.append(f"    FP  {cat:<20} {cnt:>6}  {pct:>5.1f}%")
        for cat in FN_CATS:
            cnt = r["fn_counts"][cat]
            pct = 100*cnt/fn_tot if fn_tot>0 else 0
            lines.append(f"    FN  {cat:<20} {cnt:>6}  {pct:>5.1f}%")
    lines.append("="*70)
    txt = "\n".join(lines)
    (out_path/"comparison.txt").write_text(txt, encoding="utf-8")
    print(txt)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_models = len(all_results)
        fig, axes = plt.subplots(2, n_models, figsize=(7*n_models, 10))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle(f"Analisi Errori -- {n_models} modelli ({n_images} immagini)",
                     fontsize=14, fontweight="bold")

        pal_fp = ["#9B59B6","#8FBC44","#D4AC0D","#E07B54","#E74C3C"]
        pal_fn = ["#8E44AD","#2ECC71","#95A5A6","#5DADE2","#3498DB"]

        for col, r in enumerate(all_results):
            fp_tot = sum(r["fp_counts"].values())
            fn_tot = sum(r["fn_counts"].values())
            fnr    = 100*fn_tot/max(r["n_gt_total"],1)

            ax = axes[0, col]
            vals = [r["fp_counts"][c] for c in FP_CATS]
            pcts = [100*v/fp_tot if fp_tot>0 else 0 for v in vals]
            bars = ax.barh(FP_CATS[::-1], vals[::-1], color=pal_fp[::-1], alpha=0.85)
            ax.set_title(f"{r['name']}\nFP totali: {fp_tot}", fontsize=11)
            ax.set_xlabel("Conteggio"); ax.set_ylabel("False Positives")
            ax.invert_yaxis()
            for bar, pct in zip(bars, pcts[::-1]):
                ax.text(bar.get_width()+max(vals,default=1)*0.01,
                        bar.get_y()+bar.get_height()/2,
                        f"{pct:.1f}%", va="center", fontsize=9)

            ax = axes[1, col]
            vals = [r["fn_counts"][c] for c in FN_CATS]
            pcts = [100*v/fn_tot if fn_tot>0 else 0 for v in vals]
            bars = ax.barh(FN_CATS[::-1], vals[::-1], color=pal_fn[::-1], alpha=0.85)
            ax.set_title(f"FN totali: {fn_tot}  (FNR {fnr:.1f}%)", fontsize=11)
            ax.set_xlabel("Conteggio"); ax.set_ylabel("False Negatives")
            ax.invert_yaxis()
            for bar, pct in zip(bars, pcts[::-1]):
                ax.text(bar.get_width()+max(vals,default=1)*0.01,
                        bar.get_y()+bar.get_height()/2,
                        f"{pct:.1f}%", va="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(str(out_path/"comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> Grafico: {out_path}/comparison.png")

        if n_models == 2:
            r0, r1 = all_results[0], all_results[1]
            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
            fig2.suptitle(f"delta errori: {r1['name']} vs {r0['name']}\n"
                          "(negativo = meno errori = migliore)",
                          fontsize=12, fontweight="bold")
            for ax, cats, c0, c1, lbl in [
                (axes2[0], FP_CATS, r0["fp_counts"], r1["fp_counts"], "FP"),
                (axes2[1], FN_CATS, r0["fn_counts"], r1["fn_counts"], "FN"),
            ]:
                t0 = sum(c0.values()); t1 = sum(c1.values())
                deltas = [(100*c1[c]/t1 if t1>0 else 0) -
                          (100*c0[c]/t0 if t0>0 else 0) for c in cats]
                cols   = ["#e05c5c" if d>0 else "#5b8dd9" for d in deltas]
                ax.barh(cats[::-1], deltas[::-1], color=cols[::-1], alpha=0.85)
                ax.axvline(0, color="black", lw=0.8, ls="--")
                ax.set_xlabel("delta punti percentuali")
                ax.set_title(f"delta {lbl} (%)", fontsize=11)
                ax.invert_yaxis()
                for i, v in enumerate(deltas[::-1]):
                    ax.text(v+(0.05 if v>=0 else -0.05), i,
                            f"{v:+.2f}", va="center",
                            ha="left" if v>=0 else "right", fontsize=8)
            plt.tight_layout()
            plt.savefig(str(out_path/"delta.png"), dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  -> Delta: {out_path}/delta.png")

    except ImportError:
        print("  ! matplotlib non disponibile")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_analysis(weights_list, names_list, data_dir, split,
                 out_dir, iou_thr, conf_thr):
    data_path = Path(data_dir)
    if (data_path / split / "images").exists():
        img_dir = data_path / split / "images"
        lbl_dir = data_path / split / "labels"
    elif (data_path / "images" / split).exists():
        img_dir = data_path / "images" / split
        lbl_dir = data_path / "labels" / split
    else:
        img_dir = data_path / split
        lbl_dir = data_path / split

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not img_paths:
        print(f"  Nessuna immagine trovata in {img_dir}")
        sys.exit(1)
    print(f"\n  {len(img_paths)} immagini in {img_dir}")

    all_results = []
    for weights, name in zip(weights_list, names_list):
        print(f"\nCarico modello [{name}]: {weights}")
        model = YOLO(weights)
        fp_counts, fn_counts, n_gt, n_img = analyze_one_model(
            model, name, img_paths, lbl_dir, out_path, iou_thr, conf_thr
        )
        all_results.append(dict(
            name=name, fp_counts=fp_counts, fn_counts=fn_counts,
            n_gt_total=n_gt, n_images=n_img
        ))
        print(f"  FP={sum(fp_counts.values())}  "
              f"FN={sum(fn_counts.values())}  GT={n_gt}  "
              f"FPPI={sum(fp_counts.values())/max(n_img,1):.3f}")

    make_comparison(all_results, out_path, len(img_paths))
    make_error_table_png(all_results, out_path)
    print(f"\n  -> Output in: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analisi errori FP/FN multi-modello"
    )
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--names",   nargs="+", default=None)
    parser.add_argument("--data_dir", default="data/dataset_finale")
    parser.add_argument("--split",    default="test")
    parser.add_argument("--out_dir",  default="results/error_analysis")
    parser.add_argument("--iou_thr",  type=float, default=0.5)
    parser.add_argument("--conf_thr", type=float, default=0.25)
    args = parser.parse_args()

    if args.names is None:
        args.names = [Path(w).parent.name for w in args.weights]
    if len(args.names) != len(args.weights):
        parser.error("--names deve avere lo stesso numero di elementi di --weights")

    run_analysis(
        weights_list = args.weights,
        names_list   = args.names,
        data_dir     = args.data_dir,
        split        = args.split,
        out_dir      = args.out_dir,
        iou_thr      = args.iou_thr,
        conf_thr     = args.conf_thr,
    )