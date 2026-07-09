"""
conta_sorgenti.py
=================
Conta le immagini disponibili in Roboflow e MIVIA Video
prima di costruire il dataset clean.

USO:
    python scripts/conta_sorgenti.py `
        --video_root "C:\\Users\\186337\\Desktop\\oos1\\video pepper\\video_oos_pepper\\Video"
"""

import argparse
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2


SOGLIA_BLUR_VIDEO = 100
VIDEO_STRIDE      = 3

VIDEO_NAMES = [
    "video1","video2","video3","video4","video5","video6","video7",
    "video9","video10","video11","video12","video14","video15",
    "video17","video18","video19","video20","video21","video22","video23",
]


def blur_score(path: Path) -> float | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def conta_roboflow(roboflow_dir: Path) -> dict:
    print("\n── Roboflow ───────────────────────────────────────────────")

    if not roboflow_dir.exists():
        print(f"  ❌ Non trovato: {roboflow_dir}")
        return {}

    totali    = defaultdict(int)
    con_label = defaultdict(int)

    for split_dir in sorted(roboflow_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        if not img_dir.exists():
            continue

        n_img = 0
        n_lbl = 0
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in img_dir.glob(ext):
                n_img += 1
                lbl_path = lbl_dir / img_path.with_suffix(".txt").name
                if lbl_path.exists() and lbl_path.stat().st_size > 0:
                    n_lbl += 1

        print(f"  {split_dir.name:15s}: {n_img:5d} immagini | {n_lbl:5d} con label")
        totali["img"]    += n_img
        totali["label"]  += n_lbl

    print(f"  {'TOTALE':15s}: {totali['img']:5d} immagini | {totali['label']:5d} con label")
    return dict(totali)


def conta_mivia_video(video_root: Path) -> dict:
    print("\n── MIVIA Video ────────────────────────────────────────────")
    print(f"  (stride={VIDEO_STRIDE}, soglia_blur={SOGLIA_BLUR_VIDEO})")
    print()

    if not video_root.exists():
        print(f"  ❌ Non trovato: {video_root}")
        return {}

    def num_key(p):
        ns = re.findall(r'\d+', p.stem)
        return int(ns[-1]) if ns else 0

    totali = defaultdict(int)

    print(f"  {'Video':10s} {'Orig':>6} {'Stride':>6} {'No XML':>6} "
          f"{'XML vuoti':>9} {'Blur':>6} {'OK':>6}")
    print(f"  {'-'*65}")

    for video_name in VIDEO_NAMES:
        video_dir = video_root / video_name
        if not video_dir.exists():
            print(f"  {video_name:10s} [SKIP]")
            continue

        orig_dir  = video_dir / "ImmaginiOriginali"
        xml_dir   = video_dir / "XML"
        final_dir = video_dir / "Final"

        # video1: YOLO in Final/
        if video_name == "video1" and final_dir.exists():
            imgs = sorted(final_dir.glob("*.jpg"))
            imgs = sorted(imgs, key=num_key)[::VIDEO_STRIDE]
            n_orig   = len(list(final_dir.glob("*.jpg")))
            ok = blur = no_lbl = 0
            for img_path in imgs:
                lbl_path = final_dir / img_path.with_suffix(".txt").name
                if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                    no_lbl += 1; continue
                score = blur_score(img_path)
                if score is None or score < SOGLIA_BLUR_VIDEO:
                    blur += 1; continue
                ok += 1
            print(f"  {video_name:10s} {n_orig:>6} {len(imgs):>6} {'—':>6} "
                  f"{'—':>9} {blur:>6} {ok:>6}  [YOLO]")
            totali["orig"] += n_orig
            totali["ok"]   += ok
            continue

        # video2-23: XML
        if not orig_dir.exists() or not xml_dir.exists():
            print(f"  {video_name:10s} [SKIP] cartelle mancanti")
            continue

        imgs_all = (sorted(orig_dir.glob("*.jpg")) +
                    sorted(orig_dir.glob("*.jpeg")) +
                    sorted(orig_dir.glob("*.png")))
        n_orig  = len(imgs_all)
        imgs    = sorted(imgs_all, key=num_key)[::VIDEO_STRIDE]

        ok = blur = no_xml = no_obj = 0
        for img_path in imgs:
            xml_path = xml_dir / img_path.with_suffix(".xml").name
            if not xml_path.exists():
                no_xml += 1; continue
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                has_obj = len(root.findall("object")) > 0
            except ET.ParseError:
                no_xml += 1; continue
            if not has_obj:
                no_obj += 1; continue
            score = blur_score(img_path)
            if score is None or score < SOGLIA_BLUR_VIDEO:
                blur += 1; continue
            ok += 1

        print(f"  {video_name:10s} {n_orig:>6} {len(imgs):>6} {no_xml:>6} "
              f"{no_obj:>9} {blur:>6} {ok:>6}")
        totali["orig"] += n_orig
        totali["ok"]   += ok

    print(f"  {'-'*65}")
    print(f"  {'TOTALE':10s} {totali['orig']:>6} {'':>6} {'':>6} "
          f"{'':>9} {'':>6} {totali['ok']:>6}")
    return dict(totali)


def main():
    parser = argparse.ArgumentParser(
        description="Conta immagini disponibili in Roboflow e MIVIA Video"
    )
    parser.add_argument("--roboflow_dir", default="data/raw/roboflow",
                        help="Path a data/raw/roboflow (default: data/raw/roboflow)")
    parser.add_argument("--video_root",   required=True,
                        help="Path root cartella Video MIVIA")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CONTEGGIO SORGENTI DATASET CLEAN")
    print(f"{'='*60}")

    r = conta_roboflow(Path(args.roboflow_dir))
    v = conta_mivia_video(Path(args.video_root))

    rob_ok  = r.get("label", 0)
    vid_ok  = v.get("ok", 0)
    totale  = rob_ok + vid_ok

    tr = int(totale * 0.70)
    va = int(totale * 0.15)
    te = totale - tr - va

    print(f"\n{'='*60}")
    print(f"  RIEPILOGO FINALE")
    print(f"{'='*60}")
    print(f"  Roboflow     : {rob_ok:>6,} immagini con label")
    print(f"  MIVIA Video  : {vid_ok:>6,} immagini (post-filtro)")
    print(f"  {'─'*35}")
    print(f"  TOTALE       : {totale:>6,} immagini")
    print(f"  Split atteso :")
    print(f"    train      : {tr:>6,}  (70%)")
    print(f"    val        : {va:>6,}  (15%)")
    print(f"    test       : {te:>6,}  (15%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()