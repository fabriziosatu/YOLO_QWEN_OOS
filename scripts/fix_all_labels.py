"""
scripts/fix_all_labels.py
==========================
Normalizza tutte le label in data/processed_clean/labels/ a classe 0.

Problema trovato:
  - rf_*.txt    : classe 0 ('0'=spazio vuoto) e classe 1 ('gap'=spazio vuoto)
                  → entrambe valide, rimappa tutto a classe 0
  - mivia_*.txt : classe 1 (almost_empty o formato errato) + 6 colonne
                  → forza classe 0, rimuove colonna confidence

Dopo questo script tutti i file avranno esclusivamente:
    0 xc yc w h
    (5 colonne, classe 0 = empty_shelf)

Uso:
    python scripts/fix_all_labels.py --dry_run
    python scripts/fix_all_labels.py
    python scripts/fix_all_labels.py --data_dir data/processed_clean
"""

import argparse
from pathlib import Path
from collections import Counter


def fix_label_file(lbl_path: Path, dry_run: bool) -> dict:
    """
    Corregge un singolo file label:
      - Accetta 5 colonne (formato YOLO standard) o 6 colonne (con confidence)
      - Rimappa qualsiasi classe a 0
      - Scarta box degeneri (w=0 o h=0)
      - Clamp coordinate in [0,1]

    Returns:
        dict con statistiche della correzione
    """
    with open(lbl_path) as f:
        lines = f.readlines()

    new_lines  = []
    stats      = Counter()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        n_cols = len(parts)

        if n_cols == 5:
            # Formato YOLO standard: cls xc yc w h
            try:
                cls_orig = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), \
                               float(parts[3]), float(parts[4])
            except ValueError:
                stats['malformed'] += 1
                continue

        elif n_cols == 6:
            # Formato errato con confidence: cls xc yc w h conf
            try:
                cls_orig = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), \
                               float(parts[3]), float(parts[4])
                # parts[5] = confidence → scartato
            except ValueError:
                stats['malformed'] += 1
                continue

        else:
            stats['malformed'] += 1
            continue

        # Clamp coordinate
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w  = max(0.0, min(1.0, w))
        h  = max(0.0, min(1.0, h))

        if w <= 0 or h <= 0:
            stats['degenerate'] += 1
            continue

        # Traccia la classe originale per il report
        stats[f'class_{cls_orig}'] += 1
        stats['total'] += 1

        # Scrivi sempre classe 0
        new_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    if not dry_run:
        with open(lbl_path, 'w') as f:
            f.writelines(new_lines)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Normalizza tutte le label a classe 0 (empty_shelf)"
    )
    parser.add_argument(
        '--data_dir', default='data/processed_clean',
        help='Cartella root del dataset (default: data/processed_clean)'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Mostra statistiche senza modificare i file'
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print(f"\nfix_all_labels.py")
    print(f"  data_dir : {data_dir}")
    print(f"  dry_run  : {args.dry_run}")

    global_stats = Counter()
    global_files = 0

    for split in ('train', 'val', 'test'):
        lbl_dir = data_dir / 'labels' / split
        if not lbl_dir.exists():
            print(f"\n  [{split}] cartella non trovata — saltato")
            continue

        all_files = sorted(lbl_dir.glob('*.txt'))
        if not all_files:
            print(f"\n  [{split}] nessun file trovato")
            continue

        split_stats = Counter()
        n_rf    = 0
        n_mivia = 0
        n_other = 0

        for lbl_path in all_files:
            file_stats = fix_label_file(lbl_path, dry_run=args.dry_run)
            split_stats += file_stats

            if lbl_path.name.startswith('rf_'):
                n_rf += 1
            elif lbl_path.name.startswith('mivia_'):
                n_mivia += 1
            else:
                n_other += 1

        global_stats += split_stats
        global_files += len(all_files)

        # Classi originali trovate in questo split
        orig_classes = {k: v for k, v in split_stats.items()
                        if k.startswith('class_')}

        print(f"\n  [{split}]  {len(all_files)} file  "
              f"(rf={n_rf}, mivia={n_mivia}, other={n_other})")
        print(f"    Box totali     : {split_stats['total']}")
        print(f"    Box degeneri   : {split_stats['degenerate']}")
        print(f"    Righe malformed: {split_stats['malformed']}")
        print(f"    Classi originali trovate:")
        for cls_key, count in sorted(orig_classes.items()):
            cls_num = cls_key.replace('class_', '')
            print(f"      classe {cls_num}: {count:7d} box")

    # ── Riepilogo globale ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  RIEPILOGO GLOBALE")
    print(f"{'='*55}")
    print(f"  File elaborati   : {global_files}")
    print(f"  Box totali       : {global_stats['total']}")
    print(f"  Box degeneri     : {global_stats['degenerate']}")
    print(f"  Righe malformed  : {global_stats['malformed']}")

    orig_classes = {k: v for k, v in global_stats.items()
                    if k.startswith('class_')}
    print(f"  Classi originali :")
    for cls_key, count in sorted(orig_classes.items()):
        cls_num = cls_key.replace('class_', '')
        label   = 'empty_shelf (gap)' if cls_num == '1' else \
                  'empty_shelf (0)'   if cls_num == '0' else \
                  f'classe {cls_num}'
        print(f"    {cls_num} ({label}): {count:7d} box → rimappate a 0")

    if args.dry_run:
        print(f"\n  (dry_run — nessun file modificato)")
        print(f"  Riesegui senza --dry_run per applicare")
    else:
        print(f"\n  ✓ Tutte le label normalizzate a classe 0!")
        print(f"\n  Elimina la cache YOLO prima di rilanciare il training:")
        print(f"    rm -f {data_dir}/labels/train.cache")
        print(f"    rm -f {data_dir}/labels/val.cache")
        print(f"    rm -f {data_dir}/labels/test.cache")
        print(f"\n  Verifica:")
        print(f"    head -3 {data_dir}/labels/train/rf_*.txt | head -6")


if __name__ == '__main__':
    main()