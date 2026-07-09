"""
scripts/dataset_stats.py
==========================
Restituisce statistiche dettagliate per ogni split del dataset merged:
  - Numero di immagini
  - Box empty_shelf (classe 0)
  - Box full (classi diverse da 0, se presenti)
  - Box totali
  - Media box per immagine

Uso:
  python scripts/dataset_stats.py
  python scripts/dataset_stats.py --data_dir data/merged_dataset
"""

import argparse
from pathlib import Path
from collections import Counter


def stats_split(lbl_dir: Path, img_dir: Path) -> dict:
    """Calcola le statistiche per uno split."""
    imgs = list(img_dir.glob("*.jpg")) + \
           list(img_dir.glob("*.jpeg")) + \
           list(img_dir.glob("*.png"))
    n_images = len(imgs)

    classes   = Counter()
    n_labeled = 0
    n_empty_imgs = 0   # immagini con almeno 1 box empty

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        boxes_in_file = Counter()
        with open(lbl_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cls = int(parts[0])
                        classes[cls] += 1
                        boxes_in_file[cls] += 1
                    except ValueError:
                        pass
        if boxes_in_file:
            n_labeled += 1
        if 0 in boxes_in_file:
            n_empty_imgs += 1

    n_empty = classes.get(0, 0)
    n_full  = sum(v for k, v in classes.items() if k != 0)
    n_total = sum(classes.values())

    return {
        "n_images":     n_images,
        "n_labeled":    n_labeled,
        "n_empty_imgs": n_empty_imgs,
        "box_empty":    n_empty,
        "box_full":     n_full,
        "box_total":    n_total,
        "avg_box":      round(n_total / n_images, 2) if n_images else 0,
        "classes":      dict(classes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/merged_dataset",
        help="Cartella root del dataset (default: data/merged_dataset)")
    args = parser.parse_args()

    base = Path(args.data_dir)

    print(f"\nDataset: {base.resolve()}")
    print(f"{'='*65}")

    totals = {"n_images": 0, "box_empty": 0, "box_full": 0, "box_total": 0}

    for split in ("train", "val", "test"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split

        if not img_dir.exists():
            print(f"\n  [{split}] cartella non trovata — saltato")
            continue

        s = stats_split(lbl_dir, img_dir)

        print(f"\n  [{split.upper()}]")
        print(f"    Immagini totali          : {s['n_images']:6d}")
        print(f"    Immagini con label       : {s['n_labeled']:6d}")
        print(f"    Immagini con empty_shelf : {s['n_empty_imgs']:6d}")
        print(f"    Box empty_shelf (cls 0)  : {s['box_empty']:6d}")
        print(f"    Box full (cls != 0)      : {s['box_full']:6d}")
        print(f"    Box totali               : {s['box_total']:6d}")
        print(f"    Media box/immagine       : {s['avg_box']:6.2f}")
        if s['classes']:
            print(f"    Classi trovate           : {s['classes']}")

        for k in totals:
            totals[k] += s[k]

    print(f"\n{'='*65}")
    print(f"  TOTALE (train + val + test)")
    print(f"{'='*65}")
    print(f"    Immagini totali          : {totals['n_images']:6d}")
    print(f"    Box empty_shelf (cls 0)  : {totals['box_empty']:6d}")
    print(f"    Box full (cls != 0)      : {totals['box_full']:6d}")
    print(f"    Box totali               : {totals['box_total']:6d}")
    avg = round(totals['box_total'] / totals['n_images'], 2) if totals['n_images'] else 0
    print(f"    Media box/immagine       : {avg:6.2f}")
    print()


if __name__ == "__main__":
    main()