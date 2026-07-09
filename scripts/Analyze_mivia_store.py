"""
scripts/analyze_mivia_store.py
================================
Analizza i dataset MIVIA Store e conta le classi per store.
Mappatura classi (da instances_val.json di Supermarket4):
    classe 0 → almost_empty  ← da scartare
    classe 1 → empty         ← positivi, rimappare a 0
"""

from pathlib import Path
from collections import Counter

root = Path(r'C:\Users\186337\Desktop\oos1\OOS dataset')

stores = {
    'store6  (Web Market)':          'Web Market/store6',
    'store7  (Supermarket1 Agropoli)': 'Supermarket1 - Agropoli/store7',
    'store8  (Supermarket2 Gragnano)': 'Supermarket2 - Gragnano/store8',
    'store9  (Supermarket3 Salerno)':  'Supermarket3 - Salerno/store9',
    'store10 (SKU110K MIVIA)':         'SKU110K/store10',
}

print('Store MIVIA — analisi classi')
print('Mappatura: classe 1 = empty (positivi), classe 0 = almost_empty (scartare)')
print()

total_imgs         = 0
total_empty_boxes  = 0
total_almost_boxes = 0
total_imgs_empty   = 0
total_imgs_almost  = 0
total_imgs_noann   = 0

for store_name, rel_path in stores.items():
    lbl_dir = root / rel_path / 'labels'
    img_dir = root / rel_path / 'images'

    txts = list(lbl_dir.glob('*.txt'))
    imgs = list(img_dir.glob('*'))

    classes              = Counter()
    imgs_with_empty      = 0
    imgs_with_only_almost = 0
    imgs_no_annotation   = len(imgs) - len(txts)

    for f in txts:
        file_classes = Counter()
        for line in f.read_text(encoding='utf-8', errors='replace').splitlines():
            parts = line.strip().split()
            if parts and len(parts) == 5:
                file_classes[parts[0]] += 1

        has_empty  = '1' in file_classes
        has_almost = '0' in file_classes

        if has_empty:
            imgs_with_empty += 1
        elif has_almost and not has_empty:
            imgs_with_only_almost += 1

        classes += file_classes

    n_empty  = classes.get('1', 0)
    n_almost = classes.get('0', 0)

    total_imgs         += len(imgs)
    total_empty_boxes  += n_empty
    total_almost_boxes += n_almost
    total_imgs_empty   += imgs_with_empty
    total_imgs_almost  += imgs_with_only_almost
    total_imgs_noann   += imgs_no_annotation

    print(f'  {store_name}')
    print(f'    Immagini totali             : {len(imgs)}')
    print(f'    File label                  : {len(txts)}')
    print(f'    Immagini senza label        : {imgs_no_annotation}')
    print(f'    Box empty      (cls 1 -> 0) : {n_empty:5d}  <- positivi')
    print(f'    Box almost_empty (cls 0)    : {n_almost:5d}  <- scartare')
    print(f'    Img con almeno 1 empty      : {imgs_with_empty:5d}  <- usabili come positivi')
    print(f'    Img solo almost_empty       : {imgs_with_only_almost:5d}  <- potenziali negativi')
    print()

print('=' * 55)
print('TOTALE store MIVIA')
print('=' * 55)
print(f'  Immagini totali             : {total_imgs}')
print(f'  Box empty (positivi)        : {total_empty_boxes}')
print(f'  Box almost_empty (scartate) : {total_almost_boxes}')
print(f'  Img usabili come positivi   : {total_imgs_empty}')
print(f'  Img solo almost (negativi)  : {total_imgs_almost}')
print(f'  Img senza annotazione       : {total_imgs_noann}')
print()
print('  Nota: le immagini senza annotazione e quelle con')
print('  solo almost_empty sono candidati naturali come')
print('  negativi espliciti (scaffali non completamente vuoti)')