"""
scripts/merge_datasets.py
==========================
Fonde processed_clean_cluster e roboflow_filtered in un unico dataset
con split 70/15/15 e rimappatura classe 0 -> 1 (empty_shelf).

Sorgenti:
  1. data/processed_clean_cluster/  -- 7.111 img, split esistente train/val/test
  2. data/roboflow_filtered/        -- 536 img, nessuno split (cartella flat)

Output in data/merged_dataset/:
  images/{train,val,test}/
  labels/{train,val,test}/   <- classe 1 = empty_shelf (rimappata da 0)
  data.yaml

Split: 70% train / 15% val / 15% test applicato globalmente
su tutto il pool unificato (i vecchi split vengono ignorati).

Uso:
  python scripts/merge_datasets.py
  python scripts/merge_datasets.py --dry_run
  python scripts/merge_datasets.py \
      --processed_dir data/processed_clean_cluster \
      --roboflow_dir  data/roboflow_filtered \
      --out_dir       data/merged_dataset
"""

import argparse
import random
import shutil
from pathlib import Path
from collections import Counter


SPLIT_RATIOS = (0.70, 0.15, 0.15)
IMG_SUFFIXES = {'.jpg', '.jpeg', '.png'}
SEED         = 42


# ── Rimappatura label ─────────────────────────────────────────────────────────

def remap_label_file(src_lbl: Path, dst_lbl: Path, dry_run: bool) -> int:
    """
    Legge un file .txt YOLO, rimappa classe 0 -> 1, scrive in dst_lbl.
    Scarta righe con classi diverse da 0.

    Returns:
        numero di box scritte
    """
    new_lines = []
    with open(src_lbl, encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                xc, yc, w, h = (float(p) for p in parts[1:])
            except ValueError:
                continue
            if cls != 0:
                continue
            if w <= 0 or h <= 0:
                continue
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w  = max(0.0, min(1.0, w))
            h  = max(0.0, min(1.0, h))
            new_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    if not dry_run and new_lines:
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_lbl, 'w') as f:
            f.writelines(new_lines)

    return len(new_lines)


# ── Raccolta campioni ─────────────────────────────────────────────────────────

def collect_from_split(base_dir: Path, split: str, prefix: str) -> list:
    """Raccoglie coppie (immagine, label) da una cartella split esistente."""
    img_dir = base_dir / 'images' / split
    lbl_dir = base_dir / 'labels' / split
    if not img_dir.exists():
        return []
    samples = []
    for img_path in sorted(img_dir.glob('*')):
        if img_path.suffix.lower() not in IMG_SUFFIXES:
            continue
        lbl_path = lbl_dir / img_path.with_suffix('.txt').name
        if not lbl_path.exists():
            continue
        samples.append({'img': img_path, 'lbl': lbl_path, 'prefix': prefix})
    return samples


def collect_flat(base_dir: Path, prefix: str) -> list:
    """Raccoglie coppie da una cartella flat (nessuno split)."""
    img_dir = base_dir / 'images'
    lbl_dir = base_dir / 'labels'
    if not img_dir.exists():
        return []
    samples = []
    for img_path in sorted(img_dir.glob('*')):
        if img_path.suffix.lower() not in IMG_SUFFIXES:
            continue
        lbl_path = lbl_dir / img_path.with_suffix('.txt').name
        if not lbl_path.exists():
            continue
        samples.append({'img': img_path, 'lbl': lbl_path, 'prefix': prefix})
    return samples


# ── Split ─────────────────────────────────────────────────────────────────────

def split_samples(samples: list, ratios: tuple, seed: int) -> dict:
    random.seed(seed)
    shuffled = list(samples)
    random.shuffle(shuffled)
    n       = len(shuffled)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return {
        'train': shuffled[:n_train],
        'val':   shuffled[n_train:n_train + n_val],
        'test':  shuffled[n_train + n_val:],
    }


# ── Scrittura output ──────────────────────────────────────────────────────────

def write_split(split: str, samples: list, out_dir: Path, dry_run: bool) -> dict:
    img_out = out_dir / 'images' / split
    lbl_out = out_dir / 'labels' / split
    if not dry_run:
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    n_boxes  = 0
    for s in samples:
        img_path = s['img']
        lbl_path = s['lbl']
        out_stem = f"{s['prefix']}_{img_path.stem}"
        dst_img  = img_out / f"{out_stem}{img_path.suffix}"
        dst_lbl  = lbl_out / f"{out_stem}.txt"
        if not dry_run:
            shutil.copy2(img_path, dst_img)
        n_boxes  += remap_label_file(lbl_path, dst_lbl, dry_run=dry_run)
        n_copied += 1
    return {'n_images': n_copied, 'n_boxes': n_boxes}


def write_yaml(out_dir: Path, dry_run: bool):
    content = f"""# Dataset: empty shelf detection -- merged
# Generato da merge_datasets.py
# Classe: 0 = empty_shelf

path: {out_dir.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: empty_shelf
"""
    if not dry_run:
        (out_dir / 'data.yaml').write_text(content)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', default='data/processed_clean_cluster')
    parser.add_argument('--roboflow_dir',  default='data/roboflow_filtered')
    parser.add_argument('--out_dir',       default='data/merged_dataset')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    roboflow_dir  = Path(args.roboflow_dir)
    out_dir       = Path(args.out_dir)

    print(f"\nmerge_datasets.py")
    print(f"  processed_dir : {processed_dir}")
    print(f"  roboflow_dir  : {roboflow_dir}")
    print(f"  out_dir       : {out_dir}")
    print(f"  split         : {SPLIT_RATIOS[0]*100:.0f}% / "
          f"{SPLIT_RATIOS[1]*100:.0f}% / {SPLIT_RATIOS[2]*100:.0f}%")
    print(f"  rimappatura   : classe 0 -> 1 (empty_shelf)")
    print(f"  dry_run       : {args.dry_run}")

    # ── Raccolta ──────────────────────────────────────────────────────────────
    print(f"\nRaccolta campioni...")
    all_samples = []

    if processed_dir.exists():
        for split in ('train', 'val', 'test'):
            s = collect_from_split(processed_dir, split, prefix='pcc')
            all_samples.extend(s)
            print(f"  processed_clean_cluster/{split}: {len(s)} campioni")
    else:
        print(f"  WARNING: {processed_dir} non trovata")

    if roboflow_dir.exists():
        s = collect_flat(roboflow_dir, prefix='rf')
        all_samples.extend(s)
        print(f"  roboflow_filtered:         {len(s)} campioni")
    else:
        print(f"  WARNING: {roboflow_dir} non trovata")

    if not all_samples:
        raise RuntimeError("Nessun campione trovato. Controlla i percorsi.")

    print(f"\n  Totale campioni: {len(all_samples)}")

    # Verifica classi su campione
    cls_counter = Counter()
    for s in all_samples[:200]:
        with open(s['lbl'], encoding='utf-8', errors='replace') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_counter[parts[0]] += 1
    print(f"  Classi trovate (campione 200): {dict(cls_counter)}")
    unexpected = [k for k in cls_counter if k != '0']
    if unexpected:
        print(f"  WARNING: classi inattese {unexpected} -- verranno scartate")
    else:
        print(f"  OK: solo classe 0 -> verra' rimappata a 1")

    # ── Split 70/15/15 ────────────────────────────────────────────────────────
    splits = split_samples(all_samples, SPLIT_RATIOS, seed=args.seed)

    print(f"\n  Split risultante:")
    for split, samples in splits.items():
        print(f"    {split:5s}: {len(samples):5d} ({len(samples)/len(all_samples)*100:.1f}%)")

    # ── Scrittura ─────────────────────────────────────────────────────────────
    total_stats = {}
    for split, samples in splits.items():
        stats = write_split(split, samples, out_dir, dry_run=args.dry_run)
        total_stats[split] = stats
        if not args.dry_run:
            print(f"    {split:5s}: {stats['n_images']:5d} img, "
                  f"{stats['n_boxes']:6d} box scritte")

    write_yaml(out_dir, dry_run=args.dry_run)

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    total_img = sum(s['n_images'] for s in total_stats.values())
    total_box = sum(s['n_boxes']  for s in total_stats.values())

    print(f"\n{'='*50}")
    print(f"  RIEPILOGO")
    print(f"{'='*50}")
    print(f"  Totale immagini : {total_img}")
    print(f"  Totale box      : {total_box}")
    print(f"  Classe unica    : 1 = empty_shelf")
    print(f"  train / val / test : "
          f"{total_stats['train']['n_images']} / "
          f"{total_stats['val']['n_images']} / "
          f"{total_stats['test']['n_images']}")

    if args.dry_run:
        print(f"\n  (dry_run -- nessun file scritto)")
        print(f"  Riesegui senza --dry_run per applicare")
    else:
        print(f"\n  Output    : {out_dir.resolve()}")
        print(f"  data.yaml : {out_dir}/data.yaml")

    print(f"\n  Completato!")


if __name__ == '__main__':
    main()