"""
scripts/clean_internet_data.py
================================
Rimuove dal dataset i dati provenienti da internet (SKU110K e WebMarket
scaricati online) che non devono essere usati nel training.

Cosa viene rimosso:
  data/processed_clean/negatives/sku110k/    <- generato da csv_to_negatives.py
  data/processed_clean/negatives/webmarket/  <- generato da csv_to_negatives.py

Cosa NON viene toccato:
  data/processed_clean/images/               <- positivi (roboflow + mivia video)
  data/processed_clean/labels/               <- label YOLO
  data/processed_clean/negatives/confirmed.txt  <- negativi MIVIA video (ok)

Nota: lo SKU110K e WebMarket dentro OOS dataset (MIVIA) sono dataset
      del laboratorio MIVIA e vengono gestiti separatamente da
      include_mivia_store.py.

Uso:
    python scripts/clean_internet_data.py
    python scripts/clean_internet_data.py --dry_run
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Rimuove dati internet (SKU110K/WebMarket) da processed_clean"
    )
    parser.add_argument(
        '--data_dir', default='data/processed_clean',
        help='Cartella root del dataset (default: data/processed_clean)'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Mostra cosa verrebbe rimosso senza rimuovere nulla'
    )
    args = parser.parse_args()

    neg_dir = Path(args.data_dir) / 'negatives'

    print(f"\nclean_internet_data.py")
    print(f"  data_dir : {args.data_dir}")
    print(f"  dry_run  : {args.dry_run}")

    targets = [
        neg_dir / 'sku110k',
        neg_dir / 'webmarket',
        neg_dir / 'hard_negatives_report.json',
    ]

    found_any = False
    for target in targets:
        if target.exists():
            found_any = True
            if target.is_dir():
                n_files = sum(1 for _ in target.rglob('*') if _.is_file())
                print(f"\n  {'[DRY RUN] ' if args.dry_run else ''}Rimuovo cartella: {target}")
                print(f"    ({n_files} file)")
                if not args.dry_run:
                    shutil.rmtree(target)
                    print(f"    ✓ Rimossa")
            else:
                print(f"\n  {'[DRY RUN] ' if args.dry_run else ''}Rimuovo file: {target}")
                if not args.dry_run:
                    target.unlink()
                    print(f"    ✓ Rimosso")
        else:
            print(f"\n  Non trovato (già rimosso o mai creato): {target.name}")

    if not found_any:
        print("\n  Nessun dato internet trovato — dataset già pulito.")
        return

    # Verifica cosa rimane in negatives/
    print(f"\n  Contenuto attuale di {neg_dir}:")
    if neg_dir.exists():
        for item in sorted(neg_dir.iterdir()):
            if item.is_dir():
                n = sum(1 for _ in item.rglob('*') if _.is_file())
                print(f"    {item.name}/  ({n} file)")
            else:
                print(f"    {item.name}")
    else:
        print(f"    (cartella non trovata)")

    if not args.dry_run:
        print(f"\n  ✓ Pulizia completata!")
    else:
        print(f"\n  Riesegui senza --dry_run per applicare.")


if __name__ == '__main__':
    main()