"""
plot_comparison.py  v4
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def _iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6)

def _multi_match(preds, gts, iou_thr):
    n_p=len(preds); n_g=len(gts)
    mp=[[] for _ in range(n_p)]; mg=[[] for _ in range(n_g)]
    for pi,pb in enumerate(preds):
        for gi,gb in enumerate(gts):
            v=_iou(pb,gb)
            if (iou_thr>0 and v>=iou_thr) or (iou_thr==0 and v>0):
                mp[pi].append(gi); mg[gi].append(pi)
    is_tp=[len(l)>0 for l in mp]
    matched={gi for gi,l in enumerate(mg) if len(l)>0}
    return is_tp, matched

def load_pairs(img_dir, lbl_dir):
    pairs=[]
    for p in sorted(img_dir.glob("*.jpg"))+sorted(img_dir.glob("*.png")):
        lbl=lbl_dir/(p.stem+".txt")
        if not lbl.exists(): continue
        img=Image.open(p); W,H=img.size
        gt=[]
        for line in lbl.read_text().strip().splitlines():
            parts=line.split()
            if len(parts)<5: continue
            _,cx,cy,bw,bh=map(float,parts[:5])
            gt.append([(cx-bw/2)*W,(cy-bh/2)*H,(cx+bw/2)*W,(cy+bh/2)*H])
        pairs.append({"path":p,"gt":gt})
    return pairs

def eval_yolo(yolo, pairs, conf, iou_vals):
    samples=[]
    for s in pairs:
        res=yolo(s["path"],conf=conf,verbose=False)
        boxes=res[0].boxes.xyxy.cpu().numpy().tolist() \
              if res and res[0].boxes is not None else []
        samples.append({"gt":s["gt"],"preds":boxes})
    out={}
    for iou_t in iou_vals:
        gt_tot=tp=fp=fn=0
        for s in samples:
            is_tp,matched=_multi_match(s["preds"],s["gt"],iou_t)
            gt_tot+=len(s["gt"])
            tp+=sum(is_tp); fp+=sum(not x for x in is_tp)
            fn+=sum(1 for j in range(len(s["gt"])) if j not in matched)
        out[iou_t]={"TPR":tp/gt_tot if gt_tot>0 else 0.0,
                    "FPR":fp/gt_tot if gt_tot>0 else 0.0,
                    "FNR":fn/gt_tot if gt_tot>0 else 0.0}
    return out

def load_qwen(json_path, conf, iou_vals, qwen_thrs, detectors, prompt):
    with open(json_path) as f: data=json.load(f)
    # Le chiavi nel JSON usano "aug"/"noaug", non "detector_only_aug"
    # Mappa entrambe le forme
    def det_key(det):
        return det.replace("detector_only_", "")
    out={}
    for det in detectors:
        out[det]={}
        for thr in qwen_thrs:
            out[det][thr]={}
            for iou_t in iou_vals:
                # Prova prima con il nome completo, poi con quello corto
                for dk in [det, det_key(det)]:
                    k=f"conf={conf}_iou={iou_t}_qwen={thr}_{dk}_{prompt}"
                    if k in data:
                        m=data[k]
                        out[det][thr][iou_t]={
                            "TPR":m["TPR"],
                            "FPR":m["FP"]/m["GT"] if m["GT"]>0 else 0.0,
                            "FNR":m["FNR"]}
                        break
    return out


def plot(yolo_results, qwen_results, detectors, qwen_thrs,
         iou_vals, conf, prompt, out_path):

    metrics    = ["TPR","FPR","FNR"]
    met_colors = {"TPR":"#2ecc71","FPR":"#e74c3c","FNR":"#f39c12"}

    det_short = {"detector_only_noaug":"No Aug","detector_only_aug":"Aug"}

    # Pipeline labels per asse X
    # Una pipeline = (detector, source) dove source = YOLO o Qwen_thr
    pip_labels = []
    for det in detectors:
        ds = det_short[det]
        pip_labels.append(f"{ds}\nYOLO")
        for thr in qwen_thrs:
            pip_labels.append(f"{ds}\nQwen {thr}")

    n_pip = len(pip_labels)
    bar_w = 0.18
    # posizioni gruppo: ogni pipeline ha 3 barre (TPR/FPR/FNR) affiancate
    # con un piccolo gap tra pipeline e un gap più grande tra detector
    n_det   = len(detectors)
    n_src   = 1 + len(qwen_thrs)   # YOLO + n qwen

    # Solo detector_only_aug (esclude noaug)
    detectors = ["detector_only_aug"]

    fig, axes_grid = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
    axes = [axes_grid[0][0], axes_grid[0][1], axes_grid[1][0], axes_grid[1][1]]

    fig.suptitle(
        f"YOLO-only vs YOLO+Qwen (Aug) — conf={conf}  prompt={prompt}  split=test",
        fontsize=13, fontweight="bold"
    )

    for ax, iou_t in zip(axes, iou_vals):
        x_cursor = 0.0
        xtick_pos   = []
        xtick_label = []

        for di, det in enumerate(detectors):
            if di > 0:
                x_cursor += 0.6   # gap tra detector

            sources = [("YOLO", None)] + [("Qwen", thr) for thr in qwen_thrs]

            for si, (src, thr) in enumerate(sources):
                if si > 0:
                    x_cursor += 0.05  # gap tra pipeline

                # Recupera valori
                if src == "YOLO":
                    vals = yolo_results[det].get(iou_t, {})
                else:
                    vals = qwen_results[det].get(thr, {}).get(iou_t, {})

                # Stile visivo per pipeline: border spesso e colore bordo diverso
                pip_styles = [
                    {"edgecolor":"#111111","linewidth":2.0,"alpha":1.0},
                    {"edgecolor":"#1565C0","linewidth":1.8,"alpha":0.80},
                    {"edgecolor":"#6A1B9A","linewidth":1.8,"alpha":0.65},
                    {"edgecolor":"#1B5E20","linewidth":1.8,"alpha":0.50},
                ]
                style = pip_styles[si] if si < len(pip_styles) else pip_styles[-1]

                # Disegna 3 barre affiancate (TPR, FPR, FNR)
                for mi, met in enumerate(metrics):
                    xpos = x_cursor + mi * bar_w
                    v    = vals.get(met, 0.0)
                    ax.bar(xpos, v, bar_w * 0.85,
                           color=met_colors[met],
                           edgecolor=style["edgecolor"],
                           linewidth=style["linewidth"],
                           alpha=style["alpha"],
                           zorder=3 + si)
                    if v > 0.005:
                        ax.text(xpos + bar_w*0.425, v + 0.008,
                                f"{v:.2f}", ha="center", va="bottom",
                                fontsize=7, rotation=90, color="#222",
                                zorder=4 + si)

                # Etichetta pipeline sotto
                pip_center = x_cursor + (len(metrics) * bar_w) / 2 - bar_w/2
                xtick_pos.append(pip_center)
                ds = det_short[det]
                lbl = f"{ds}\nYOLO" if src=="YOLO" else f"{ds}\nQwen {thr}"
                xtick_label.append(lbl)

                x_cursor += len(metrics) * bar_w

            # Separatore verticale tra detector
            if di < n_det-1:
                ax.axvline(x_cursor + 0.3, color="#888",
                           linewidth=1.2, linestyle="--", alpha=0.5)

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_label, fontsize=7.5, ha="center", linespacing=1.3)
        ax.set_ylim(0, 1.35)
        ax.set_ylabel("Valore metrica", fontsize=9)
        ax.set_title(f"IoU = {iou_t}", fontsize=11, fontweight="bold")
        ax.set_xlim(-0.3, x_cursor + 0.3)
        ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.tick_params(axis="x", pad=6)

    # Legenda: metriche (colore) + pipeline (bordo)
    met_patches = [mpatches.Patch(facecolor=c, label=m)
                   for m, c in met_colors.items()]

    pip_names_leg = ["YOLO only"] + [f"Qwen thr={t}" for t in qwen_thrs]
    pip_styles_leg = [
        {"edgecolor":"#111111","linewidth":2.0,"alpha":1.0},
        {"edgecolor":"#1565C0","linewidth":1.8,"alpha":0.80},
        {"edgecolor":"#6A1B9A","linewidth":1.8,"alpha":0.65},
        {"edgecolor":"#1B5E20","linewidth":1.8,"alpha":0.50},
    ]
    pip_patches = [
        mpatches.Patch(facecolor="#aaaaaa",
                       edgecolor=s["edgecolor"],
                       linewidth=s["linewidth"],
                       alpha=s["alpha"],
                       label=n)
        for n, s in zip(pip_names_leg, pip_styles_leg)
    ]

    axes[0].legend(handles=met_patches + pip_patches,
                   loc="upper right", fontsize=8.5,
                   framealpha=0.92, ncol=1,
                   title="Legenda", title_fontsize=9)

    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noaug_weights", required=True)
    parser.add_argument("--aug_weights",   required=True)
    parser.add_argument("--qwen_json",     required=True)
    parser.add_argument("--qwen_thrs",     nargs="+", type=float, default=[0.3,0.35,0.4])
    parser.add_argument("--conf",          type=float, default=0.25)
    parser.add_argument("--iou",           nargs="+", type=float, default=[0.0,0.25,0.5,0.75])
    parser.add_argument("--prompt",        default="context")
    parser.add_argument("--out_dir",       required=True)
    parser.add_argument("--images",        required=True)
    parser.add_argument("--labels",        required=True)
    parser.add_argument("--max_images",    type=int, default=None)
    args = parser.parse_args()

    from ultralytics import YOLO as _Y
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(Path(args.images), Path(args.labels))
    if args.max_images: pairs = pairs[:args.max_images]
    print(f"Immagini: {len(pairs)}")

    detectors = ["detector_only_noaug","detector_only_aug"]
    weights   = {"detector_only_noaug": args.noaug_weights,
                 "detector_only_aug":   args.aug_weights}

    print("Valuto YOLO-only...")
    yolo_results = {}
    for det in detectors:
        print(f"  {det}...")
        yolo_results[det] = eval_yolo(_Y(weights[det]), pairs, args.conf, args.iou)

    print("Carico risultati Qwen...")
    qwen_results = load_qwen(args.qwen_json, args.conf, args.iou,
                             args.qwen_thrs, detectors, args.prompt)

    print("Genero grafico...")
    plot(yolo_results, qwen_results, detectors, args.qwen_thrs,
         args.iou, args.conf, args.prompt, out_dir/"bar_comparison2.png")

    print(f"\nSalvato in: {out_dir}")

if __name__ == "__main__":
    main()