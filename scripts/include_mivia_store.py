"""
scripts/include_mivia_store.py
================================
Aggiunge i dataset MIVIA Store a data/processed_clean/ come positivi,
applicando la rimappatura delle classi:
    classe 1 (empty)        → classe 0 (empty_shelf) ✓
    classe 0 (almost_empty) → SCARTATA               ✗

Le immagini con SOLO box almost_empty (nessuna empty) vengono
salvate in negatives/mivia_store_almost.txt come negativi naturali.

Sorgenti incluse (tutti dataset del laboratorio MIVIA, non internet):
    store6  → Web Market (MIVIA)
    store7  → Supermarket1 - Agropoli
    store8  → Supermarket2 - Gragnano
    store9  → Supermarket3 - Salerno
    store10 → SKU110K (MIVIA)

Split: le immagini di ogni store vengono distribuite in
train/val/test mantenendo proporzioni 75/15/10.

Uso:
    python scripts/include_mivia_store.py
    python scripts/include_mivia_store.py --dry_run
    python scripts/include_mivia_store.py --oos_dir "C:/path/to/OOS dataset"
"""

import argparse
import json
import random
import shutil
from pathlib import Path


# ── Configurazione ────────────────────────────────────────────────────────────

STORES = {
    'store6':  'Web Market/store6',
    'store7':  'Supermarket1 - Agropoli/store7',
    'store8':  'Supermarket2 - Gragnano/store8',
    'store9':  'Supermarket3 - Salerno/store9',
    'store10': 'SKU110K/store10',
}

SPLIT_RATIOS = (0.75, 0.15, 0.10)
IMG_SUFFIXES = {'.jpg', '.jpeg', '.png'}
SEED         = 42


# ── Rimappatura label ─────────────────────────────────────────────────────────

def remap_label_file(lbl_path: Path) -> tuple[list[str], bool, bool]:
    """
    Legge un file label MIVIA Store e applica la rimappatura.

    Returns:
        (new_lines, has_empty, only_almost)
        new_lines   : righe YOLO corrette (classe 0, solo empty)
        has_empty   : True se c'è almeno una box empty
        only_almost : True se ci sono solo box almost_empty (nessuna empty)
    """
    new_lines   = []
    has_empty   = False
    has_almost  = False

    with open(lbl_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls  = int(parts[0])
                xc, yc, w, h = (float(p) for p in parts[1:])
            except ValueError:
                continue

            if w <= 0 or h <= 0:
                continue

            if cls == 1:
                # empty → accetta, rimappa a 0
                has_empty = True
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                w  = max(0.0, min(1.0, w))
                h  = max(0.0, min(1.0, h))
                new_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            elif cls == 0:
                # almost_empty → scarta silenziosamente
                has_almost = True

    only_almost = has_almost and not has_empty
    return new_lines, has_empty, only_almost


# ── Elaborazione singolo store ────────────────────────────────────────────────

def process_store(
    store_name: str,
    store_dir:  Path,
    dry_run:    bool,
) -> dict:
    """
    Elabora un singolo store e raccoglie i campioni classificati.

    Returns:
        dict con positives, almost_only, no_label
    """
    img_dir = store_dir / 'images'
    lbl_dir = store_dir / 'labels'

    positives   = []   # (img_path, new_label_lines)
    almost_only = []   # img_path — solo almost_empty
    no_label    = []   # img_path — nessuna annotazione

    all_imgs = sorted(
        [p for p in img_dir.glob('*') if p.suffix.lower() in IMG_SUFFIXES]
    )

    for img_path in all_imgs:
        lbl_path = lbl_dir / img_path.with_suffix('.txt').name

        if not lbl_path.exists():
            no_label.append(img_path)
            continue

        new_lines, has_empty, only_almost = remap_label_file(lbl_path)

        if has_empty:
            positives.append((img_path, new_lines))
        elif only_almost:
            almost_only.append(img_path)
        else:
            no_label.append(img_path)

    print(f"  {store_name:35s}  "
          f"pos={len(positives):4d}  "
          f"almost_only={len(almost_only):4d}  "
          f"no_label={len(no_label):4d}")

    return {
        'store_name': store_name,
        'positives':  positives,
        'almost_only': almost_only,
        'no_label':   no_label,
    }


# ── Split e scrittura ─────────────────────────────────────────────────────────

def write_store_outputs(
    all_results:   list,
    out_root:      Path,
    split_ratios:  tuple,
    dry_run:       bool,
) -> dict:
    """
    Distribuisce i positivi in train/val/test e scrive immagini + label.
    Salva anche la lista dei negativi naturali (almost_only).
    """
    if not dry_run:
        for split in ('train', 'val', 'test'):
            (out_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (out_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
        (out_root / 'negatives').mkdir(parents=True, exist_ok=True)

    stats = {
        'per_store':    [],
        'split_counts': {'train': 0, 'val': 0, 'test': 0},
        'total_positives': 0,
        'total_almost_only': 0,
        'total_no_label': 0,
    }

    almost_only_paths = []

    random.seed(SEED)

    for res in all_results:
        store_name = res['store_name']
        positives  = res['positives']

        # Shuffle per evitare che lo split sia ordinato per nome file
        random.shuffle(positives)

        n       = len(positives)
        n_train = int(n * split_ratios[0])
        n_val   = int(n * split_ratios[1])

        split_samples = {
            'train': positives[:n_train],
            'val':   positives[n_train:n_train + n_val],
            'test':  positives[n_train + n_val:],
        }

        store_prefix = store_name.split()[0]   # 'store6', 'store7', ecc.

        for split, samples in split_samples.items():
            img_out = out_root / 'images' / split
            lbl_out = out_root / 'labels' / split

            for img_path, new_lines in samples:
                out_stem = f"mivia_{store_prefix}_{img_path.stem}"
                dst_img  = img_out / f"{out_stem}{img_path.suffix}"
                dst_lbl  = lbl_out / f"{out_stem}.txt"

                if not dry_run:
                    shutil.copy2(img_path, dst_img)
                    with open(dst_lbl, 'w') as f:
                        f.writelines(new_lines)

            stats['split_counts'][split] += len(samples)

        # Raccogli quasi-negativi
        almost_only_paths.extend(res['almost_only'])

        stats['total_positives']   += n
        stats['total_almost_only'] += len(res['almost_only'])
        stats['total_no_label']    += len(res['no_label'])

        stats['per_store'].append({
            'store':       store_name,
            'positives':   n,
            'almost_only': len(res['almost_only']),
            'no_label':    len(res['no_label']),
            'split': {
                'train': len(split_samples['train']),
                'val':   len(split_samples['val']),
                'test':  len(split_samples['test']),
            }
        })

    # Scrivi lista negativi naturali (almost_only)
    if not dry_run and almost_only_paths:
        out_file = out_root / 'negatives' / 'mivia_store_almost.txt'
        with open(out_file, 'w') as f:
            for p in almost_only_paths:
                f.write(str(p.resolve()) + '\n')

    stats['total_negatives_natural'] = len(almost_only_paths)
    return stats


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggiunge MIVIA Store a processed_clean come positivi"
    )
    parser.add_argument(
        '--oos_dir',
        default=r'C:\Users\186337\Desktop\oos1\OOS dataset',
        help='Cartella root OOS dataset MIVIA'
    )
    parser.add_argument(
        '--out_dir', default='data/processed_clean',
        help='Cartella di output (default: data/processed_clean)'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Mostra statistiche senza copiare nulla'
    )
    args = parser.parse_args()

    oos_root = Path(args.oos_dir)
    out_root = Path(args.out_dir)

    print(f"\ninclude_mivia_store.py")
    print(f"  oos_dir  : {oos_root}")
    print(f"  out_dir  : {out_root}")
    print(f"  dry_run  : {args.dry_run}")
    print(f"\n  Rimappatura classi:")
    print(f"    classe 1 (empty)        → classe 0 (empty_shelf) ✓")
    print(f"    classe 0 (almost_empty) → SCARTATA               ✗")

    if not oos_root.exists():
        raise FileNotFoundError(f"oos_dir non trovata: {oos_root}")

    print(f"\n  {'store':35s}  {'pos':>5}  {'almost_only':>11}  {'no_label':>8}")
    print(f"  {'-'*65}")

    all_results = []
    for store_name, rel_path in STORES.items():
        store_dir = oos_root / rel_path
        if not store_dir.exists():
            print(f"  {store_name:35s}  NON TROVATO — saltato")
            continue
        result = process_store(store_name, store_dir, dry_run=args.dry_run)
        all_results.append(result)

    stats = write_store_outputs(
        all_results,
        out_root     = out_root,
        split_ratios = SPLIT_RATIOS,
        dry_run      = args.dry_run,
    )

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RIEPILOGO MIVIA STORE")
    print(f"{'='*60}")

    print(f"\n  Per store:")
    for s in stats['per_store']:
        print(f"    {s['store']:10s}: pos={s['positives']:4d}  "
              f"(train={s['split']['train']} val={s['split']['val']} "
              f"test={s['split']['test']})  "
              f"almost_only={s['almost_only']}  no_label={s['no_label']}")

    print(f"\n  Positivi aggiunti:")
    for split in ('train', 'val', 'test'):
        print(f"    {split:5s}: {stats['split_counts'][split]:5d}")
    print(f"    TOTALE: {stats['total_positives']}")

    print(f"\n  Negativi naturali (solo almost_empty):")
    print(f"    {stats['total_almost_only']:5d} immagini")
    print(f"    → salvati in negatives/mivia_store_almost.txt")
    print(f"    → usati come negativi espliciti per SigLIP")

    print(f"\n  Immagini senza annotazione: {stats['total_no_label']} (scartate)")

    if not args.dry_run:
        info_path = out_root / 'mivia_store_info.json'
        info_path.write_text(json.dumps(stats, indent=2, default=str))
        print(f"\n  Report: {info_path}")
        print(f"\n  ✓ MIVIA Store aggiunto al dataset!")
    else:
        print(f"\n  (dry_run — nessun file scritto)")
        print(f"  Riesegui senza --dry_run per applicare.")


if __name__ == '__main__':
    main()