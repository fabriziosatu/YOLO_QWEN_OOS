"""
scripts/prepare_dataset.py
===========================
Unisce tutte le sorgenti di positivi in data/processed_clean/ con
split train/val/test coerenti, e verifica che i negativi siano pronti.

Sorgenti positivi:
  1. data/raw/roboflow/   — già splittato in train/valid/test con label YOLO
  2. data/processed_clean/ — frame video_oos_pepper già convertiti da xml_to_yolo.py

Sorgenti negativi (già su disco, non vengono spostati):
  data/processed_clean/negatives/confirmed.txt   — XML vuoti video_oos_pepper
  data/processed_clean/negatives/sku110k/        — hard negatives SKU110K
  data/processed_clean/negatives/webmarket/      — hard negatives WebMarket

Output finale in data/processed_clean/:
  images/{train,val,test}/   <- tutti i positivi uniti
  labels/{train,val,test}/   <- label YOLO corrispondenti
  negatives/                 <- già presente, viene solo verificata
  data.yaml                  <- config per YOLO training (Fase 1)
  dataset_info.json          <- riepilogo completo per la tesi

Uso:
  python scripts/prepare_dataset.py
  python scripts/prepare_dataset.py --dry_run
  python scripts/prepare_dataset.py --roboflow_dir data/raw/roboflow
"""

import argparse
import json
import shutil
from pathlib import Path


# ── Configurazione ────────────────────────────────────────────────────────────

ROBOFLOW_SPLIT_MAP = {
    "train": "train",
    "val":   "valid",   # roboflow usa "valid" non "val"
    "test":  "test",
}

IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}


# ── Copia immagini + label ────────────────────────────────────────────────────

def copy_split(
    src_img_dir,
    src_lbl_dir,
    dst_img_dir,
    dst_lbl_dir,
    prefix,
    dry_run,
):
    """
    Copia tutte le coppie (immagine, label) da src a dst,
    aggiungendo un prefisso al nome file per evitare collisioni.

    Returns:
        (n_copied, n_missing_label)
    """
    if not dry_run:
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_copied  = 0
    n_missing = 0

    for img_path in sorted(src_img_dir.glob("*")):
        if img_path.suffix.lower() not in IMG_SUFFIXES:
            continue

        lbl_path = src_lbl_dir / img_path.with_suffix(".txt").name
        if not lbl_path.exists():
            n_missing += 1
            continue

        out_stem = f"{prefix}_{img_path.stem}"
        dst_img  = dst_img_dir / f"{out_stem}{img_path.suffix}"
        dst_lbl  = dst_lbl_dir / f"{out_stem}.txt"

        if not dry_run:
            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)

        n_copied += 1

    return n_copied, n_missing


# ── Verifica negativi ─────────────────────────────────────────────────────────

def verify_negatives(neg_dir):
    """
    Verifica che tutte le sorgenti di negativi siano presenti e
    conta quante immagini sono disponibili per il training di Fase 2.
    """
    results = {}

    # Negativi confermati (XML vuoti da video_oos_pepper)
    confirmed = neg_dir / "confirmed.txt"
    if confirmed.exists():
        lines = [l.strip() for l in confirmed.read_text().splitlines() if l.strip()]
        results["confirmed"] = len(lines)
    else:
        results["confirmed"] = 0
        print(f"  ⚠ confirmed.txt non trovato — esegui xml_to_yolo.py prima")

    # Hard negatives SKU110K
    sku_dir = neg_dir / "sku110k"
    if sku_dir.exists():
        sku_txts = list(sku_dir.rglob("*.txt"))
        results["sku110k"] = len(sku_txts)
    else:
        results["sku110k"] = 0
        print(f"  ⚠ negatives/sku110k/ non trovato — esegui csv_to_negatives.py prima")

    # Hard negatives WebMarket
    web_dir = neg_dir / "webmarket"
    if web_dir.exists():
        web_txts = list(web_dir.glob("*.txt"))
        results["webmarket"] = len(web_txts)
    else:
        results["webmarket"] = 0
        print(f"  ⚠ negatives/webmarket/ non trovato — esegui csv_to_negatives.py prima")

    results["total"] = (results["confirmed"] +
                        results["sku110k"] +
                        results["webmarket"])
    return results


# ── Scrivi data.yaml ──────────────────────────────────────────────────────────

def write_yaml(out_dir, dry_run):
    """
    Genera data.yaml per il training YOLO (Fase 1).
    Fase 2 usa build_dataloaders() e non ha bisogno di questo file.
    """
    content = f"""# Dataset: empty shelf detection
# Generato da prepare_dataset.py
# Usato da YOLO nella Fase 1 del training

path: {out_dir.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: empty_shelf
"""
    if not dry_run:
        (out_dir / "data.yaml").write_text(content)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unisce roboflow e video_oos_pepper in processed_clean"
    )
    parser.add_argument(
        "--roboflow_dir", default="data/raw/roboflow",
        help="Cartella roboflow con train/valid/test (default: data/raw/roboflow)"
    )
    parser.add_argument(
        "--out_dir", default="data/processed_clean",
        help="Cartella di output (default: data/processed_clean)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Mostra statistiche senza copiare nulla"
    )
    args = parser.parse_args()

    rf_dir  = Path(args.roboflow_dir)
    out_dir = Path(args.out_dir)
    neg_dir = out_dir / "negatives"

    print(f"\nprepare_dataset.py")
    print(f"  roboflow_dir : {rf_dir}")
    print(f"  out_dir      : {out_dir}")
    print(f"  dry_run      : {args.dry_run}")

    if not rf_dir.exists():
        raise FileNotFoundError(f"roboflow_dir non trovata: {rf_dir}")

    stats = {
        "splits": {"train": {}, "val": {}, "test": {}},
        "negatives": {},
    }

    # ── Elabora ogni split ────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {'split':6s}  {'sorgente':20s}  {'copiati':>8}  {'no_label':>8}")
    print(f"{'─'*60}")

    for split, rf_split in ROBOFLOW_SPLIT_MAP.items():

        dst_img = out_dir / "images" / split
        dst_lbl = out_dir / "labels" / split

        split_total   = 0
        split_missing = 0

        # Sorgente 1: roboflow
        rf_img = rf_dir / rf_split / "images"
        rf_lbl = rf_dir / rf_split / "labels"

        if rf_img.exists() and rf_lbl.exists():
            n, m = copy_split(rf_img, rf_lbl, dst_img, dst_lbl,
                              prefix="rf", dry_run=args.dry_run)
            print(f"  {split:6s}  {'roboflow':20s}  {n:8d}  {m:8d}")
            split_total   += n
            split_missing += m
        else:
            print(f"  {split:6s}  roboflow/{rf_split} NON TROVATO — saltato")

        # Sorgente 2: video_oos_pepper (già in processed_clean da xml_to_yolo.py)
        # Sono già nella destinazione corretta con prefisso videoN_ — li contiamo
        existing_imgs = list(dst_img.glob("video*.jpg")) if dst_img.exists() else []
        existing_lbls = list(dst_lbl.glob("video*.txt")) if dst_lbl.exists() else []
        n_video = min(len(existing_imgs), len(existing_lbls))

        print(f"  {split:6s}  {'video_oos_pepper':20s}  {n_video:8d}  {'(già presenti)':>8}")
        split_total += n_video

        stats["splits"][split] = {
            "roboflow":         split_total - n_video,
            "video_oos_pepper": n_video,
            "total":            split_total,
            "missing_labels":   split_missing,
        }

    # ── Verifica negativi ─────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Verifica negativi in {neg_dir}")
    print(f"{'─'*60}")

    neg_stats = verify_negatives(neg_dir)
    stats["negatives"] = neg_stats

    print(f"  confirmed.txt  : {neg_stats['confirmed']:6d} frame")
    print(f"  sku110k/       : {neg_stats['sku110k']:6d} immagini")
    print(f"  webmarket/     : {neg_stats['webmarket']:6d} immagini")
    print(f"  TOTALE negativi: {neg_stats['total']:6d}")

    # ── Scrivi data.yaml ──────────────────────────────────────────────────────
    write_yaml(out_dir, dry_run=args.dry_run)

    # ── Riepilogo finale ──────────────────────────────────────────────────────
    total_pos = sum(s["total"] for s in stats["splits"].values())
    total_neg = neg_stats["total"]
    n_train_pos = stats["splits"]["train"]["total"]

    print(f"\n{'='*60}")
    print(f"  RIEPILOGO DATASET FINALE")
    print(f"{'='*60}")
    print(f"\n  POSITIVI (empty_shelf annotati):")
    for split, s in stats["splits"].items():
        print(f"    {split:5s}: {s['total']:5d}  "
              f"(roboflow={s['roboflow']}, video={s['video_oos_pepper']})")
    print(f"    TOTALE: {total_pos}")

    print(f"\n  NEGATIVI (per SigLIP — Fase 2):")
    print(f"    confirmed (XML vuoti) : {neg_stats['confirmed']:6d}")
    print(f"    SKU110K hard negatives: {neg_stats['sku110k']:6d}")
    print(f"    WebMarket hard neg.   : {neg_stats['webmarket']:6d}")
    print(f"    TOTALE                : {total_neg:6d}")

    print(f"\n  Bilanciamento per Fase 2:")
    print(f"    Positivi train        : {n_train_pos}")
    print(f"    Negativi disponibili  : {total_neg}")
    print(f"    Negativi per epoca    : {n_train_pos}  (campionamento 1:1)")
    print(f"    Epoche per vedere tutti i negativi: "
          f"~{total_neg // n_train_pos + 1}")

    stats["totals"] = {
        "positives":      total_pos,
        "negatives":      total_neg,
        "train_positives": n_train_pos,
        "balance_ratio":  round(total_neg / total_pos, 2),
    }

    if not args.dry_run:
        info_path = out_dir / "dataset_info.json"
        info_path.write_text(json.dumps(stats, indent=2, default=str))
        print(f"\n  data.yaml    : {out_dir}/data.yaml")
        print(f"  dataset_info : {info_path}")

    print(f"\n  ✓ Dataset pronto!")


if __name__ == "__main__":
    main()