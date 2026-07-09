"""
scripts/xml_to_yolo.py
=======================
Converte le annotazioni Pascal VOC XML del dataset MIVIA Video
in formato YOLO (.txt) e produce un report dettagliato.

Problemi della versione precedente — tutti corretti qui:
  1. PREFIX ERRATO: img_path.parent.parent.name restituiva 'Video' invece
     di 'videoN' quando le cartelle ImmaginiOriginali/XML erano alla radice.
     Fix: si usa esplicitamente video_info['video_name'].
  2. CLASSE ERRATA: i file .txt di ImmaginiTXT avevano classe 1 e 6 colonne.
     Fix: lo script legge SOLO gli XML, mai i .txt preesistenti.
  3. ALMOST_EMPTY non filtrata: la classe '0 almost empty' veniva inclusa.
     Fix: filtro esplicito su nome classe — solo '1 empty' (e varianti)
     viene accettato, tutto il resto scartato.
  4. 6 COLONNE: alcuni file precedenti avevano cls xc yc w h conf.
     Fix: non applicabile alla nuova conversione (si parte da XML puliti).

Classi nel dataset MIVIA Video:
  '1 empty'       → empty_shelf → classe YOLO 0  ✓ accettata
  '0 almost empty'→ almost_empty → SCARTATA       ✗ esclusa (prof)

Struttura attesa per ogni video:
    videoN/
        ImmaginiOriginali/   ← frame .jpg
        XML/                 ← annotazioni .xml (Pascal VOC)

Tre categorie di frame:
  POSITIVO  XML con almeno un <object> valido (empty_shelf)
  NEGATIVO  XML presente ma vuoto — scaffale pieno confermato
  AMBIGUO   Nessun XML — frame non esaminato, scartato

Output in data/processed_clean/:
    images/{train,val,test}/   ← .jpg dei frame positivi
    labels/{train,val,test}/   ← .txt YOLO: 0 xc yc w h (5 colonne)
    negatives/confirmed.txt    ← percorsi assoluti dei frame negativi
    negatives/ambiguous.txt    ← percorsi assoluti degli ambigui
    report.json                ← statistiche per video

Uso:
    python scripts/xml_to_yolo.py --video_root "C:/path/to/Video"
    python scripts/xml_to_yolo.py --video_root "C:/path/to/Video" --dry_run
    python scripts/xml_to_yolo.py --video_root "C:/path/to/Video" --sample_every 3
"""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


# ── Configurazione ────────────────────────────────────────────────────────────

TARGET_CLASS = 0          # unica classe output: empty_shelf
SPLIT_RATIOS = (0.75, 0.15, 0.10)
IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Nomi delle cartelle immagini e XML nei video MIVIA
IMG_FOLDER_NAMES = ["ImmaginiOriginali", "Final"]
XML_FOLDER_NAMES = ["XML"]   # ImmaginiTXT escluso — contiene .txt pre-esistenti
                              # in formato non standard che causavano il bug


# ── Filtro classi ─────────────────────────────────────────────────────────────

def is_empty_shelf(name: str) -> bool:
    """
    Restituisce True solo se il nome della classe corrisponde a empty_shelf.

    Classi accettate  : '1 empty', 'empty', 'empty_shelf', 'gap', ecc.
    Classi rifiutate  : '0 almost empty', 'almost_empty', 'almost', ecc.

    La logica è: se contiene 'almost' → rifiuta.
                 Se contiene 'empty' o 'gap' → accetta.
                 Altrimenti → rifiuta (comportamento conservativo).
    """
    name_lower = name.lower().strip()

    # Rifiuta esplicitamente almost_empty in tutte le varianti
    if 'almost' in name_lower:
        return False

    # Accetta varianti di empty e gap
    if 'empty' in name_lower or 'gap' in name_lower:
        return True

    return False


# ── Conversione Pascal VOC → YOLO ─────────────────────────────────────────────

def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Converte bbox da Pascal VOC (pixel assoluti) a YOLO (normalizzato).
    Returns: (xc, yc, w, h) in [0, 1]
    """
    xc = ((xmin + xmax) / 2) / img_w
    yc = ((ymin + ymax) / 2) / img_h
    w  = (xmax - xmin) / img_w
    h  = (ymax - ymin) / img_h

    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))

    return xc, yc, w, h


def parse_xml(xml_path: Path):
    """
    Legge un XML Pascal VOC e classifica il frame.

    Returns:
        ('positive', boxes) — almeno una box empty_shelf valida
        ('negative', [])    — XML esaminato, nessun oggetto valido
        ('error',    [])    — XML malformato
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return 'error', []

    size  = root.find('size')
    img_w = int(size.findtext('width',  '640')) if size is not None else 640
    img_h = int(size.findtext('height', '480')) if size is not None else 480

    objects = root.findall('object')

    # XML senza oggetti → frame esaminato e confermato vuoto
    if not objects:
        return 'negative', []

    boxes        = []
    n_almost     = 0
    n_unknown    = 0

    for obj in objects:
        name    = obj.findtext('name', '').strip()
        bndbox  = obj.find('bndbox')

        if bndbox is None:
            continue

        # Filtro classe
        if not is_empty_shelf(name):
            if 'almost' in name.lower():
                n_almost += 1
            else:
                n_unknown += 1
            continue

        try:
            xmin = int(float(bndbox.findtext('xmin', '0')))
            ymin = int(float(bndbox.findtext('ymin', '0')))
            xmax = int(float(bndbox.findtext('xmax', str(img_w))))
            ymax = int(float(bndbox.findtext('ymax', str(img_h))))
        except ValueError:
            continue

        if xmax <= xmin or ymax <= ymin:
            continue

        xc, yc, w, h = voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        boxes.append((xc, yc, w, h))

    # Se tutti gli oggetti erano almost_empty → frame negativo
    # (non ci sono spazi vuoti veri, solo zone quasi vuote)
    if not boxes:
        return 'negative', []

    return 'positive', boxes


# ── Scoperta struttura video ──────────────────────────────────────────────────

def find_video_folders(video_root: Path) -> list:
    """
    Scansiona video_root cercando coppie (img_dir, xml_dir).

    FIX rispetto alla versione precedente:
    - Cerca SOLO cartelle che si chiamano videoN (contengono 'video')
    - Evita di prendere ImmaginiOriginali/XML alla radice come video separato
    - Usa video_info['video_name'] = candidate.name (sempre corretto)
    """
    found = []

    for candidate in sorted(video_root.iterdir()):
        if not candidate.is_dir():
            continue

        # Accetta solo cartelle che sembrano video (video1, video2, ecc.)
        # Esclude cartelle come ImmaginiOriginali, Final, XML, .vscode
        name_lower = candidate.name.lower()
        if not ('video' in name_lower or name_lower[0].isdigit()):
            continue
        if candidate.name.startswith('.'):
            continue

        # Cerca img_dir
        img_dir = None
        for fname in IMG_FOLDER_NAMES:
            d = candidate / fname
            if d.is_dir():
                img_dir = d
                break

        if img_dir is None:
            # Prova la root del video stesso
            imgs = list(candidate.glob('*.jpg')) + list(candidate.glob('*.png'))
            if imgs:
                img_dir = candidate

        if img_dir is None:
            continue

        # Cerca xml_dir — SOLO nella cartella XML, mai in ImmaginiTXT
        xml_dir = None
        for fname in XML_FOLDER_NAMES:
            for base in [candidate, img_dir]:
                d = base / fname
                if d.is_dir() and list(d.glob('*.xml')):
                    xml_dir = d
                    break
            if xml_dir:
                break

        if xml_dir is None:
            continue

        found.append({
            'video_name': candidate.name,   # es. 'video2' — SEMPRE corretto
            'img_dir':    img_dir,
            'xml_dir':    xml_dir,
        })

    return found


# ── Elaborazione singolo video ────────────────────────────────────────────────

def process_video(video_info: dict, sample_every: int = 1) -> dict:
    """
    Elabora un singolo video classificando ogni frame.
    """
    img_dir  = video_info['img_dir']
    xml_dir  = video_info['xml_dir']
    vid_name = video_info['video_name']

    all_imgs = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES],
        key=lambda p: _sort_key(p.stem)
    )
    all_imgs = all_imgs[::sample_every]

    positives = []
    negatives = []
    ambiguous = []
    errors    = []

    for img_path in all_imgs:
        xml_path = xml_dir / f"{img_path.stem}.xml"

        if not xml_path.exists():
            ambiguous.append({'img': img_path, 'xml': None})
            continue

        category, boxes = parse_xml(xml_path)

        if category == 'positive':
            positives.append({
                'img':        img_path,
                'xml':        xml_path,
                'boxes':      boxes,
                'video_name': vid_name,   # ← passato esplicitamente
            })
        elif category == 'negative':
            negatives.append({'img': img_path, 'xml': xml_path})
        else:
            errors.append({'img': img_path, 'xml': xml_path})

    print(f"  {vid_name:20s}  frames={len(all_imgs):5d}  "
          f"pos={len(positives):4d}  neg={len(negatives):4d}  "
          f"amb={len(ambiguous):4d}  err={len(errors):2d}")

    return {
        'video_name': vid_name,
        'total':      len(all_imgs),
        'positives':  positives,
        'negatives':  negatives,
        'ambiguous':  ambiguous,
        'errors':     errors,
    }


def _sort_key(stem: str) -> int:
    digits = ''.join(c for c in stem if c.isdigit())
    return int(digits) if digits else 0


# ── Split train/val/test ──────────────────────────────────────────────────────

def assign_splits(samples: list, ratios: tuple) -> dict:
    n       = len(samples)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return {
        'train': samples[:n_train],
        'val':   samples[n_train:n_train + n_val],
        'test':  samples[n_train + n_val:],
    }


# ── Scrittura output ──────────────────────────────────────────────────────────

def write_outputs(
    all_video_results,
    out_root,
    copy_images  = True,
    dry_run      = False,
    split_ratios = SPLIT_RATIOS,
) -> dict:
    """
    Scrive immagini, label YOLO, lista negativi e report.

    FIX rispetto alla versione precedente:
    - Il nome del file usa s['video_name'] passato esplicitamente
      invece di img_path.parent.parent.name (che era sbagliato)
    - Prefisso: 'video2_image42' invece di 'Video_image42'
    - Formato label: sempre 5 colonne '0 xc yc w h'
    """
    all_positives = []
    for res in all_video_results:
        all_positives.extend(res['positives'])

    splits = assign_splits(all_positives, split_ratios)

    global_stats = {
        'total_frames_examined': 0,
        'total_positives':  0,
        'total_negatives':  0,
        'total_ambiguous':  0,
        'total_errors':     0,
        'split_counts':     {},
        'per_video':        [],
    }

    if not dry_run:
        for split in ('train', 'val', 'test'):
            (out_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (out_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
        (out_root / 'negatives').mkdir(parents=True, exist_ok=True)

    # ── Scrivi positivi ───────────────────────────────────────────────────────
    for split, samples in splits.items():
        img_out = out_root / 'images' / split
        lbl_out = out_root / 'labels' / split

        for s in samples:
            img_path   = s['img']
            boxes      = s['boxes']
            video_name = s['video_name']   # FIX: usa il nome passato, non parent.parent

            # Prefisso corretto: 'video2_image42'
            out_stem = f"{video_name}_{img_path.stem}"
            out_img  = img_out / f"{out_stem}{img_path.suffix}"
            out_lbl  = lbl_out / f"{out_stem}.txt"

            if not dry_run:
                if copy_images:
                    shutil.copy2(img_path, out_img)

                # FIX: sempre 5 colonne, sempre classe 0
                with open(out_lbl, 'w') as f:
                    for (xc, yc, w, h) in boxes:
                        f.write(f"{TARGET_CLASS} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        global_stats['split_counts'][split] = len(samples)

    global_stats['total_positives'] = len(all_positives)

    # ── Scrivi negativi e ambigui ─────────────────────────────────────────────
    all_negatives = []
    all_ambiguous = []
    for res in all_video_results:
        all_negatives.extend(res['negatives'])
        all_ambiguous.extend(res['ambiguous'])

    global_stats['total_negatives'] = len(all_negatives)
    global_stats['total_ambiguous'] = len(all_ambiguous)

    if not dry_run:
        with open(out_root / 'negatives' / 'confirmed.txt', 'w') as f:
            for s in all_negatives:
                f.write(str(s['img'].resolve()) + '\n')

        with open(out_root / 'negatives' / 'ambiguous.txt', 'w') as f:
            for s in all_ambiguous:
                f.write(str(s['img'].resolve()) + '\n')

    # ── Statistiche per video ─────────────────────────────────────────────────
    for res in all_video_results:
        n_tot = res['total']
        n_pos = len(res['positives'])
        n_neg = len(res['negatives'])
        n_amb = len(res['ambiguous'])
        n_err = len(res['errors'])

        global_stats['total_frames_examined'] += n_tot
        global_stats['total_errors']          += n_err

        global_stats['per_video'].append({
            'video':        res['video_name'],
            'total':        n_tot,
            'positives':    n_pos,
            'negatives':    n_neg,
            'ambiguous':    n_amb,
            'errors':       n_err,
            'coverage_pct': round((n_pos + n_neg) / n_tot * 100, 1) if n_tot else 0,
        })

    if not dry_run:
        with open(out_root / 'report.json', 'w') as f:
            json.dump(global_stats, f, indent=2)

    return global_stats


# ── YAML per YOLO ─────────────────────────────────────────────────────────────

def write_yaml(out_root: Path, dry_run: bool = False):
    content = f"""# Dataset: MIVIA Video — empty shelf detection
# Generato da xml_to_yolo.py

path: {out_root.resolve()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: empty_shelf
"""
    if not dry_run:
        (out_root / 'data.yaml').write_text(content)
    else:
        print("\n--- data.yaml (dry run) ---")
        print(content)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Converti annotazioni Pascal VOC MIVIA Video → YOLO"
    )
    parser.add_argument('--video_root', required=True,
        help='Cartella radice con video1/, video2/, ecc.')
    parser.add_argument('--out_dir', default='data/processed_clean',
        help='Cartella di output (default: data/processed_clean)')
    parser.add_argument('--sample_every', type=int, default=3,
        help='Prendi 1 frame ogni N (default: 3)')
    parser.add_argument('--no_copy_images', action='store_true',
        help='Scrivi solo le label, non copiare le immagini')
    parser.add_argument('--dry_run', action='store_true',
        help='Mostra statistiche senza scrivere nulla')
    parser.add_argument('--split_ratios', nargs=3, type=float,
        default=list(SPLIT_RATIOS), metavar=('TRAIN', 'VAL', 'TEST'),
        help='Proporzioni split (default: 0.75 0.15 0.10)')
    args = parser.parse_args()

    video_root   = Path(args.video_root)
    out_root     = Path(args.out_dir)
    split_ratios = tuple(args.split_ratios)

    if not video_root.exists():
        raise FileNotFoundError(f"video_root non trovata: {video_root}")

    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios devono sommare a 1.0")

    print(f"\nMIVIA Video → YOLO converter (v2 — fixed)")
    print(f"  video_root  : {video_root}")
    print(f"  out_dir     : {out_root}")
    print(f"  sample_every: ogni {args.sample_every} frame")
    print(f"  split       : train={split_ratios[0]} val={split_ratios[1]} "
          f"test={split_ratios[2]}")
    print(f"  dry_run     : {args.dry_run}")
    print(f"\n  Filtro classi:")
    print(f"    ✓ accettate : 'empty', 'gap' e varianti → classe 0")
    print(f"    ✗ scartate  : 'almost empty' e varianti")

    video_folders = find_video_folders(video_root)

    if not video_folders:
        raise RuntimeError(
            f"Nessuna cartella video trovata in {video_root}.\n"
            f"Verifica che esistano sottocartelle videoN/ con "
            f"ImmaginiOriginali/ e XML/."
        )

    print(f"\nTrovati {len(video_folders)} video.\n")
    print(f"  {'video':20s}  {'frames':>6}  {'pos':>5}  {'neg':>5}  "
          f"{'amb':>5}  {'err':>4}")
    print(f"  {'-'*58}")

    all_results = []
    for vf in video_folders:
        result = process_video(vf, sample_every=args.sample_every)
        all_results.append(result)

    stats = write_outputs(
        all_results,
        out_root     = out_root,
        copy_images  = not args.no_copy_images,
        dry_run      = args.dry_run,
        split_ratios = split_ratios,
    )

    write_yaml(out_root, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print(f"  RISULTATO CONVERSIONE")
    print(f"{'='*60}")
    print(f"  Frame totali esaminati      : {stats['total_frames_examined']:6d}")
    print(f"  Positivi (empty_shelf)      : {stats['total_positives']:6d}")
    print(f"  Negativi (confermati pieni) : {stats['total_negatives']:6d}")
    print(f"  Ambigui  (nessun XML)       : {stats['total_ambiguous']:6d}")
    print(f"  Errori   (XML malformati)   : {stats['total_errors']:6d}")
    print(f"\n  Split positivi:")
    for split in ('train', 'val', 'test'):
        n = stats['split_counts'].get(split, 0)
        print(f"    {split:5s}: {n:5d} immagini")

    if not args.dry_run:
        print(f"\n  Output in        : {out_root.resolve()}")
        print(f"  Negativi conf.   : {out_root}/negatives/confirmed.txt")
        print(f"  Report           : {out_root}/report.json")
        print(f"  YOLO config      : {out_root}/data.yaml")
        print(f"\n  Prossimo passo — fix label roboflow:")
        print(f"  python scripts/fix_all_labels.py")

    print(f"\n  ✓ Conversione completata!")


if __name__ == '__main__':
    main()