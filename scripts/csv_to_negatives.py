"""
csv_to_negatives.py
====================
Converte le annotazioni CSV di SKU110K e WebMarket in file .txt
per immagine, da usare come hard negatives durante il training.

Formato input CSV:
  WebMarket  : filename, xmin, ymin, xmax, ymax, class, width, height  (con header)
  SKU110K    : filename, x1, y1, x2, y2, class, width, height          (senza header)

Formato output .txt per immagine (coordinate normalizzate [0,1]):
  x1_norm y1_norm x2_norm y2_norm
  x1_norm y1_norm x2_norm y2_norm
  ...

Una riga per ogni prodotto annotato nell'immagine.
Le coordinate sono normalizzate rispetto alle dimensioni originali
dell'immagine indicate nel CSV.

Nota: il formato è volutamente DIVERSO da YOLO (che usa xc yc w h).
Queste non sono label di empty_shelf — sono posizioni di prodotti
usate dalla JointPipeline per costruire hard negative ROI per SigLIP.

Struttura output:
  data/processed_clean/negatives/
  ├── sku110k/
  │   ├── train_0.txt
  │   ├── train_1.txt
  │   └── ...
  └── webmarket/
      ├── db1.txt
      ├── db2.txt
      └── ...

Uso:
  # Converti entrambi i dataset
  python csv_to_negatives.py \\
      --sku_dir   /path/to/sku110k \\
      --web_dir   /path/to/webmarket \\
      --out_dir   data/processed_clean/negatives

  # Solo WebMarket
  python csv_to_negatives.py --web_dir /path/to/webmarket --out_dir ...

  # Dry run (statistiche senza scrivere)
  python csv_to_negatives.py --sku_dir ... --web_dir ... --out_dir ... --dry_run
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


# ── Parsing CSV ───────────────────────────────────────────────────────────────

def parse_webmarket_csv(csv_path: Path) -> dict[str, list]:
    """
    Legge db1_data.csv (WebMarket).
    Ha header: filename, xmin, ymin, xmax, ymax, class, width, height

    Returns:
        dict {filename: [(x1n, y1n, x2n, y2n), ...]} coordinate normalizzate
    """
    boxes = defaultdict(list)
    errors = 0

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fname = row['filename'].strip()
                x1 = int(row['xmin']);  y1 = int(row['ymin'])
                x2 = int(row['xmax']);  y2 = int(row['ymax'])
                w  = int(row['width']); h  = int(row['height'])
            except (ValueError, KeyError):
                errors += 1
                continue

            if x2 <= x1 or y2 <= y1 or w <= 0 or h <= 0:
                errors += 1
                continue

            x1n = max(0.0, min(1.0, x1 / w))
            y1n = max(0.0, min(1.0, y1 / h))
            x2n = max(0.0, min(1.0, x2 / w))
            y2n = max(0.0, min(1.0, y2 / h))

            boxes[fname].append((x1n, y1n, x2n, y2n))

    if errors:
        print(f"    ⚠ {errors} righe saltate (valori non validi)")

    return dict(boxes)


def parse_sku110k_csv(csv_path: Path) -> dict[str, list]:
    """
    Legge annotations_train/val/test.csv (SKU110K).
    NON ha header: filename, x1, y1, x2, y2, class, width, height

    Returns:
        dict {filename: [(x1n, y1n, x2n, y2n), ...]} coordinate normalizzate
    """
    boxes = defaultdict(list)
    errors = 0

    with open(csv_path, newline='', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 8:
                errors += 1
                continue
            try:
                fname = parts[0].strip()
                x1 = int(parts[1]); y1 = int(parts[2])
                x2 = int(parts[3]); y2 = int(parts[4])
                w  = int(parts[6]); h  = int(parts[7])
            except ValueError:
                errors += 1
                continue

            if x2 <= x1 or y2 <= y1 or w <= 0 or h <= 0:
                errors += 1
                continue

            x1n = max(0.0, min(1.0, x1 / w))
            y1n = max(0.0, min(1.0, y1 / h))
            x2n = max(0.0, min(1.0, x2 / w))
            y2n = max(0.0, min(1.0, y2 / h))

            boxes[fname].append((x1n, y1n, x2n, y2n))

    if errors:
        print(f"    ⚠ {errors} righe saltate (valori non validi)")

    return dict(boxes)


# ── Scrittura .txt per immagine ───────────────────────────────────────────────

def write_negative_txts(
    boxes_by_image: dict[str, list],
    out_dir:        Path,
    dry_run:        bool = False,
) -> dict:
    """
    Scrive un file .txt per ogni immagine con le coordinate normalizzate
    delle box prodotto.

    Formato output per riga:
        x1_norm y1_norm x2_norm y2_norm

    Args:
        boxes_by_image: {filename: [(x1n,y1n,x2n,y2n), ...]}
        out_dir:        cartella di destinazione
        dry_run:        se True non scrive nulla

    Returns:
        statistiche: n_images, n_boxes, n_empty (immagini senza box)
    """
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_boxes  = 0
    n_empty  = 0

    for filename, box_list in sorted(boxes_by_image.items()):
        stem    = Path(filename).stem
        out_txt = out_dir / f"{stem}.txt"

        if not box_list:
            n_empty += 1
            continue

        n_images += 1
        n_boxes  += len(box_list)

        if not dry_run:
            with open(out_txt, 'w') as f:
                for (x1n, y1n, x2n, y2n) in box_list:
                    f.write(f"{x1n:.6f} {y1n:.6f} {x2n:.6f} {y2n:.6f}\n")

    return {
        'n_images': n_images,
        'n_boxes':  n_boxes,
        'n_empty':  n_empty,
        'avg_boxes_per_image': round(n_boxes / n_images, 1) if n_images else 0,
    }


# ── Elaborazione SKU110K (3 split) ───────────────────────────────────────────

def process_sku110k(sku_dir: Path, out_dir: Path, dry_run: bool) -> dict:
    """
    Processa tutti e tre i CSV di SKU110K (train/val/test).
    Ogni split finisce in una sottocartella separata per mantenere
    la struttura originale (utile se vuoi campionare solo da train).
    """
    csv_files = {
        'train': sku_dir / 'annotations_train.csv',
        'val':   sku_dir / 'annotations_val.csv',
        'test':  sku_dir / 'annotations_test.csv',
    }

    total_stats = {'splits': {}}

    for split, csv_path in csv_files.items():
        if not csv_path.exists():
            print(f"    ⚠ Non trovato: {csv_path.name} — saltato")
            continue

        print(f"  Parsing {csv_path.name}...")
        boxes = parse_sku110k_csv(csv_path)

        split_out = out_dir / 'sku110k' / split
        stats = write_negative_txts(boxes, split_out, dry_run=dry_run)

        print(f"    immagini: {stats['n_images']:6d}  "
              f"box: {stats['n_boxes']:8d}  "
              f"avg box/img: {stats['avg_boxes_per_image']:.1f}")

        total_stats['splits'][split] = stats

    total_stats['total_images'] = sum(
        s['n_images'] for s in total_stats['splits'].values()
    )
    total_stats['total_boxes'] = sum(
        s['n_boxes'] for s in total_stats['splits'].values()
    )
    return total_stats


# ── Elaborazione WebMarket ────────────────────────────────────────────────────

def process_webmarket(web_dir: Path, out_dir: Path, dry_run: bool) -> dict:
    """
    Processa db1_data.csv di WebMarket.
    Tutte le 300 immagini finiscono in negatives/webmarket/.
    """
    csv_path = web_dir / 'db1_data.csv'

    if not csv_path.exists():
        # Cerca il CSV con qualsiasi nome
        csvs = list(web_dir.glob('*.csv'))
        if not csvs:
            raise FileNotFoundError(f"Nessun CSV trovato in {web_dir}")
        csv_path = csvs[0]
        print(f"  Trovato CSV: {csv_path.name}")

    print(f"  Parsing {csv_path.name}...")
    boxes = parse_webmarket_csv(csv_path)

    wm_out = out_dir / 'webmarket'
    stats  = write_negative_txts(boxes, wm_out, dry_run=dry_run)

    print(f"    immagini: {stats['n_images']:6d}  "
          f"box: {stats['n_boxes']:8d}  "
          f"avg box/img: {stats['avg_boxes_per_image']:.1f}")

    return stats


# ── Report README ─────────────────────────────────────────────────────────────

def write_readme(out_dir: Path, sku_stats: dict, web_stats: dict):
    """
    Scrive un README nella cartella negatives/ che documenta
    il contenuto e il formato per chi legge il progetto.
    """
    readme = """# negatives/ — Hard negative mining data

Questa cartella contiene le informazioni sui negativi usati
per il training di SigLIP nella JointPipeline.

## Struttura

### confirmed.txt
Percorsi assoluti di frame estratti da video_oos_pepper i cui file XML
sono stati esaminati dall'annotatore e risultano vuoti (nessun <object>).
Sono negativi espliciti e affidabili.

### sku110k/{train,val,test}/*.txt
Un file per ogni immagine SKU110K. Ogni riga contiene le coordinate
normalizzate [0,1] di una bounding box di prodotto:

    x1_norm y1_norm x2_norm y2_norm

Queste box vengono usate dalla JointPipeline come hard negatives:
le ROI vengono campionate esattamente sopra i prodotti annotati,
così SigLIP deve imparare a distinguere "spazio vuoto" da "prodotto".

Riferimento: Goldman et al., "Precise Detection in Densely Packed Scenes",
CVPR 2019.

### webmarket/*.txt
Stesso formato di sku110k. Annotazioni dal dataset WebMarket
(db1_data.csv), scaffali di supermercato con prodotti annotati.

## Formato coordinate
Tutte le coordinate sono normalizzate rispetto alle dimensioni
originali dell'immagine (indicate nel CSV sorgente).
NON è formato YOLO (che usa xc yc w h) — sono x1 y1 x2 y2
perché rappresentano posizioni di prodotti, non label di empty_shelf.

## Bilanciamento durante il training
Il DataLoader campiona ad ogni epoca un sottoinsieme di hard negatives
pari al numero di positivi nel training set, per mantenere il rapporto 1:1.
Le immagini rimanenti ruotano nelle epoche successive.
"""
    (out_dir / 'README.md').write_text(readme)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Converti CSV SKU110K e WebMarket in .txt per hard negative mining"
    )
    parser.add_argument('--sku_dir', default=None,
        help='Cartella SKU110K con annotations_train/val/test.csv e images/')
    parser.add_argument('--web_dir', default=None,
        help='Cartella WebMarket con db1_data.csv e images/')
    parser.add_argument('--out_dir', default='data/processed_clean/negatives',
        help='Cartella di output (default: data/processed_clean/negatives)')
    parser.add_argument('--dry_run', action='store_true',
        help='Mostra statistiche senza scrivere nulla')
    args = parser.parse_args()

    if not args.sku_dir and not args.web_dir:
        parser.error("Specifica almeno uno tra --sku_dir e --web_dir")

    out_dir = Path(args.out_dir)

    print(f"\nCSV → Hard Negatives converter")
    print(f"  out_dir : {out_dir}")
    print(f"  dry_run : {args.dry_run}")

    sku_stats = None
    web_stats = None

    # ── SKU110K ───────────────────────────────────────────────────────────────
    if args.sku_dir:
        sku_dir = Path(args.sku_dir)
        if not sku_dir.exists():
            raise FileNotFoundError(f"sku_dir non trovata: {sku_dir}")
        print(f"\n[SKU110K] {sku_dir}")
        sku_stats = process_sku110k(sku_dir, out_dir, dry_run=args.dry_run)

    # ── WebMarket ─────────────────────────────────────────────────────────────
    if args.web_dir:
        web_dir = Path(args.web_dir)
        if not web_dir.exists():
            raise FileNotFoundError(f"web_dir non trovata: {web_dir}")
        print(f"\n[WebMarket] {web_dir}")
        web_stats = process_webmarket(web_dir, out_dir, dry_run=args.dry_run)

    # ── Report finale ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RISULTATO")
    print(f"{'='*60}")

    if sku_stats:
        print(f"\n  SKU110K:")
        for split, s in sku_stats['splits'].items():
            print(f"    {split:5s}: {s['n_images']:5d} immagini, "
                  f"{s['n_boxes']:8d} box, "
                  f"{s['avg_boxes_per_image']:.1f} box/img")
        print(f"    TOTALE: {sku_stats['total_images']:5d} immagini, "
              f"{sku_stats['total_boxes']:8d} box")

    if web_stats:
        print(f"\n  WebMarket:")
        print(f"    TOTALE: {web_stats['n_images']:5d} immagini, "
              f"{web_stats['n_boxes']:8d} box, "
              f"{web_stats['avg_boxes_per_image']:.1f} box/img")

    # Salva report JSON e README
    if not args.dry_run:
        report = {}
        if sku_stats:
            report['sku110k'] = sku_stats
        if web_stats:
            report['webmarket'] = web_stats

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'hard_negatives_report.json').write_text(
            __import__('json').dumps(report, indent=2)
        )
        write_readme(out_dir, sku_stats, web_stats)
        print(f"\n  Output scritto in: {out_dir.resolve()}")
        print(f"  Report           : {out_dir}/hard_negatives_report.json")
        print(f"  README           : {out_dir}/README.md")

    print(f"\n  ✓ Conversione completata!")


if __name__ == '__main__':
    main()