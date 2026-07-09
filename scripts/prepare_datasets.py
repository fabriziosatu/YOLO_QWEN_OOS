"""
prepare_datasets.py
====================
Converte WebMarket e SKU-110K nel formato YOLO e costruisce
data/processed/ con split train/val/test.

Struttura attesa in data/raw/:
  data/raw/webmarket/
      db1_data.csv          ← file CSV delle annotazioni
      images/               ← cartella con tutte le immagini .jpg

  data/raw/sku110k/
      annotations_train.csv
      annotations_val.csv
      annotations_test.csv
      images/               ← cartella con TUTTE le immagini
          train_0.jpg, train_1.jpg ...
          val_0.jpg, val_1.jpg ...
          test_0.jpg, test_1.jpg ...

  data/raw/roboflow/        ← (opzionale) già in formato YOLO
      images/train/, images/val/, images/test/
      labels/train/, labels/val/, labels/test/

Output: data/processed/
  images/train/, images/val/, images/test/
  labels/train/, labels/val/, labels/test/

Esegui con:
  python scripts/prepare_datasets.py
"""

import os
import csv
import random
import shutil
from pathlib import Path
from PIL import Image
from collections import defaultdict

# ── Configurazione ────────────────────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# test = 1 - TRAIN_RATIO - VAL_RATIO = 0.15

SEED = 42

# Per SKU-110K: quante immagini massimo usare (sono 11k, ne bastano ~500-1000)
# Metti None per usarle tutte (sconsigliato, molto lento)
SKU110K_MAX_IMAGES = 800

# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
counters = {"train": 0, "val": 0, "test": 0}


def ensure_dirs():
    for split in ["train", "val", "test"]:
        (PROCESSED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    print("✓ Cartelle output create.")


def assign_split() -> str:
    """Assegna casualmente train/val/test rispettando i ratio."""
    r = random.random()
    if r < TRAIN_RATIO:
        return "train"
    elif r < TRAIN_RATIO + VAL_RATIO:
        return "val"
    else:
        return "test"


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """
    Converte da coordinate assolute (x1,y1,x2,y2) in pixel
    a formato YOLO normalizzato (xc, yc, w, h) in [0,1].
    """
    x1 = max(0.0, float(x1))
    y1 = max(0.0, float(y1))
    x2 = min(float(img_w), float(x2))
    y2 = min(float(img_h), float(y2))

    w  = x2 - x1
    h  = y2 - y1

    if w <= 0 or h <= 0:
        return None  # box degenere, scartala

    xc = (x1 + w / 2) / img_w
    yc = (y1 + h / 2) / img_h
    wn = w / img_w
    hn = h / img_h

    # Clamp per sicurezza
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    wn = min(max(wn, 0.0), 1.0)
    hn = min(max(hn, 0.0), 1.0)

    return xc, yc, wn, hn


def save_sample(img_path: Path, boxes_yolo: list, split: str, stem: str) -> bool:
    """
    Copia l'immagine e scrive il file label .txt nella cartella del split.
    Restituisce True se l'operazione è andata a buon fine.
    """
    if not img_path.exists():
        return False
    if not boxes_yolo:
        return False

    # Destinazioni
    dst_img = PROCESSED_DIR / "images" / split / f"{stem}.jpg"
    dst_lbl = PROCESSED_DIR / "labels" / split / f"{stem}.txt"

    # Copia / converti immagine in JPEG
    try:
        img = Image.open(img_path).convert("RGB")
        img.save(dst_img, "JPEG", quality=95)
    except Exception as e:
        print(f"  ⚠ Errore immagine {img_path}: {e}")
        return False

    # Scrivi label (classe sempre 0 = empty shelf)
    with open(dst_lbl, "w") as f:
        for box in boxes_yolo:
            xc, yc, w, h = box
            f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    counters[split] += 1
    return True


# ── WebMarket ─────────────────────────────────────────────────────────────────
def process_webmarket():
    """
    Formato CSV con header:
    filename, xmin, ymin, xmax, ymax, class, width, height
    """
    src      = RAW_DIR / "webmarket"
    csv_file = src / "db1_data.csv"
    img_dir  = src / "images"

    if not csv_file.exists():
        print("⚠ WebMarket: db1_data.csv non trovato — skip.")
        return

    # Raggruppa le box per immagine
    annotations = defaultdict(list)
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname  = row["filename"].strip()
            x1     = float(row["xmin"])
            y1     = float(row["ymin"])
            x2     = float(row["xmax"])
            y2     = float(row["ymax"])
            img_w  = int(row["width"])
            img_h  = int(row["height"])
            box = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            if box:
                annotations[fname].append(box)

    count_ok = 0
    for i, (filename, boxes) in enumerate(annotations.items()):
        img_path = img_dir / filename
        split    = assign_split()
        stem     = f"webmarket_{i:05d}"
        if save_sample(img_path, boxes, split, stem):
            count_ok += 1

    print(f"✓ WebMarket: {count_ok}/{len(annotations)} immagini processate.")


# ── SKU-110K ──────────────────────────────────────────────────────────────────
def process_sku110k():
    """
    Formato CSV senza header:
    filename, x1, y1, x2, y2, class, img_w, img_h

    NOTA: SKU-110K annota i prodotti sugli scaffali, non gli spazi vuoti.
    Lo usiamo per insegnare a YOLO la struttura generale degli scaffali.
    Limitiamo il numero di immagini con SKU110K_MAX_IMAGES.
    """
    src     = RAW_DIR / "sku110k"
    img_dir = src / "images"

    csv_files = {
        "train": src / "annotations_train.csv",
        "val":   src / "annotations_val.csv",
        "test":  src / "annotations_test.csv",
    }

    # Controlla che almeno un file esista
    found = [p for p in csv_files.values() if p.exists()]
    if not found:
        print("⚠ SKU-110K: nessun CSV trovato — skip.")
        return

    # Raggruppa box per immagine (da tutti i CSV insieme)
    annotations = defaultdict(list)
    for csv_path in found:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 8:
                    continue
                fname = row[0].strip()
                try:
                    x1, y1, x2, y2 = float(row[1]), float(row[2]), \
                                      float(row[3]), float(row[4])
                    img_w, img_h    = int(row[6]), int(row[7])
                except ValueError:
                    continue
                box = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                if box:
                    annotations[fname].append(box)

    # Sottocampionamento
    all_filenames = list(annotations.keys())
    random.shuffle(all_filenames)
    if SKU110K_MAX_IMAGES is not None:
        all_filenames = all_filenames[:SKU110K_MAX_IMAGES]
    print(f"  SKU-110K: utilizzo {len(all_filenames)} immagini su {len(annotations)} totali.")

    count_ok = 0
    for i, filename in enumerate(all_filenames):
        boxes    = annotations[filename]
        img_path = img_dir / filename
        split    = assign_split()
        stem     = f"sku110k_{i:05d}"
        if save_sample(img_path, boxes, split, stem):
            count_ok += 1

    print(f"✓ SKU-110K: {count_ok}/{len(all_filenames)} immagini processate.")


# ── Roboflow (già in formato YOLO) ────────────────────────────────────────────
def process_roboflow():
    """
    Gestisce la struttura: data/raw/roboflow/{split}/images/
    """
    src = RAW_DIR / "roboflow"
    if not src.exists():
        print("⚠ Roboflow: cartella non trovata — skip.")
        return

    count_ok = 0
    # Cerchiamo in tutte le sottocartelle (train, test, valid)
    # Usiamo 'valid' invece di 'val' se Roboflow lo ha nominato così
    for split_dir in src.iterdir():
        if not split_dir.is_dir():
            continue
            
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        
        if not img_dir.exists():
            continue

        for img_file in img_dir.glob("*.jpg"):
            # Il file label ha lo stesso nome dell'immagine ma estensione .txt
            lbl_file = lbl_dir / f"{img_file.stem}.txt"

            if not lbl_file.exists():
                continue

            # Leggi le box
            boxes = []
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            # Prendiamo solo le coordinate (scartiamo la classe originale di Roboflow 
                            # perché la forziamo a 0 per la nostra tesi)
                            boxes.append(tuple(float(p) for p in parts[1:]))
                        except ValueError:
                            continue

            # Assegniamo un nuovo split casuale per mescolarlo agli altri dataset
            split = assign_split()
            stem  = f"roboflow_{img_file.stem}"
            if save_sample(img_file, boxes, split, stem):
                count_ok += 1

    print(f"✓ Roboflow: {count_ok} immagini processate.")


# ── Statistiche finali ────────────────────────────────────────────────────────
def print_stats():
    print("\n── Riepilogo dataset processato ──────────────────────────")
    total = 0
    for split in ["train", "val", "test"]:
        imgs = list((PROCESSED_DIR / "images" / split).glob("*.jpg"))
        lbls = list((PROCESSED_DIR / "labels" / split).glob("*.txt"))
        # Conta box totali
        n_boxes = 0
        for lbl in lbls:
            with open(lbl) as f:
                n_boxes += sum(1 for line in f if line.strip())
        print(f"  {split:5s}: {len(imgs):5d} immagini | {n_boxes:7d} box")
        total += len(imgs)
    print(f"  {'TOT':5s}: {total:5d} immagini")
    print("──────────────────────────────────────────────────────────")

    # Crea il file dataset.yaml necessario per YOLO
    yaml_path = PROCESSED_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {PROCESSED_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n")
        f.write("test:  images/test\n")
        f.write("\n")
        f.write("nc: 1\n")
        f.write("names: ['empty_shelf']\n")
    print(f"✓ File dataset.yaml creato in {yaml_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Preparazione dataset ===\n")
    ensure_dirs()

    print("\n[1/3] WebMarket")
    process_webmarket()

    print("\n[2/3] SKU-110K")
    process_sku110k()

    print("\n[3/3] Roboflow")
    process_roboflow()

    print_stats()
    print("\n✓ Fatto! Dataset pronto in data/processed/")