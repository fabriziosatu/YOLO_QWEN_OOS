"""
scripts/count_classes.py
=========================
Conta la distribuzione delle classi nel dataset.

Uso:
  python scripts/count_classes.py
  python scripts/count_classes.py --data_dir data/oos_ridotto
  python scripts/count_classes.py --data_dir data/processed
"""

import argparse
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/oos_ridotto")
    return p.parse_args()


def count_split(lbl_dir: Path, split: str):
    counts     = defaultdict(int)
    n_images   = 0
    n_empty    = 0   # immagini senza box

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        n_images += 1
        has_box = False
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cls = int(parts[0])
                        counts[cls] += 1
                        has_box = True
                    except ValueError:
                        continue
        if not has_box:
            n_empty += 1

    return counts, n_images, n_empty


def main():
    args     = parse_args()
    base     = Path(args.data_dir)
    splits   = ["train", "val", "test"]
    names    = {0: "empty_shelf", 1: "full_shelf"}

    print(f"\nDataset: {base}")
    print("=" * 55)

    total_counts   = defaultdict(int)
    total_images   = 0

    for split in splits:
        lbl_dir = base / "labels" / split
        if not lbl_dir.exists():
            print(f"  [{split}] directory non trovata, skip")
            continue

        counts, n_images, n_empty = count_split(lbl_dir, split)
        total_images += n_images
        for cls, n in counts.items():
            total_counts[cls] += n

        total_boxes = sum(counts.values())
        print(f"\n  [{split}] — {n_images} immagini, {total_boxes} box totali")
        for cls in sorted(counts.keys()):
            name = names.get(cls, f"classe_{cls}")
            pct  = 100 * counts[cls] / total_boxes if total_boxes > 0 else 0
            print(f"    classe {cls} ({name:12s}): {counts[cls]:5d} box  ({pct:.1f}%)")
        if n_empty > 0:
            print(f"    immagini senza box: {n_empty}")

    print(f"\n{'='*55}")
    print(f"  TOTALE — {total_images} immagini")
    total_boxes = sum(total_counts.values())
    for cls in sorted(total_counts.keys()):
        name = names.get(cls, f"classe_{cls}")
        pct  = 100 * total_counts[cls] / total_boxes if total_boxes > 0 else 0
        print(f"    classe {cls} ({name:12s}): {total_counts[cls]:5d} box  ({pct:.1f}%)")

    # Valutazione
    print(f"\n{'='*55}")
    c0 = total_counts.get(0, 0)
    c1 = total_counts.get(1, 0)
    if c1 == 0:
        print("  ✓ Dataset pulito — solo empty_shelf annotati")
        print("    Il bug nel _load_labels non ha avuto impatto.")
    elif c1 < c0 * 0.1:
        print(f"  ⚠ Poche box full_shelf ({c1} su {total_boxes} totali)")
        print("    Impatto limitato — fine-tuning dai pesi esistenti consigliato.")
    else:
        print(f"  ✗ Box full_shelf significative ({c1} su {total_boxes} totali, {100*c1/total_boxes:.1f}%)")
        print("    Il modello ha imparato a rilevare anche i prodotti pieni.")
        print("    → Rifare Fase 1 con il dataset corretto (fix in dataset.py)")
    print()


if __name__ == "__main__":
    main()
    