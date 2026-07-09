"""
prepara_oos_ridotto.py
======================
Costruisce il dataset ridotto OOS nella cartella  data/oos_ridotto/
senza toccare data/processed/ (dataset attuale in uso).

REGOLA FONDAMENTALE:
  Vengono incluse SOLO le immagini che hanno un file .txt di label
  corrispondente e NON vuoto. I video privati e le immagini store
  senza annotazione vengono esclusi completamente.

Struttura label attesa per gli store:
  <OOS dataset>/
  └── Grocery Products/
      └── store1/
          ├── images/   ← immagini .jpg
          └── labels/   ← .txt YOLO (stessa cartella parent di images/)

Struttura creata:
    data/
    ├── processed/          ← INVARIATO (dataset attuale)
    └── oos_ridotto/        ← NUOVO
        ├── dataset.yaml
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/

USO:
    python prepara_oos_ridotto.py ^
        --store_list     sharp_paths_Store_OOS.txt ^
        --target         5000 ^
        --project_root   C:\\Users\\orlan\\Desktop\\tesi_progetto

OPZIONI:
    --target        Numero totale immagini (default 5000)
    --seed          Seed riproducibilità (default 42)
"""

import argparse
import math
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


# ── Parametri campionamento ───────────────────────────────────────────────────

TRAIN_RATIO          = 0.70
VAL_RATIO            = 0.15
TEST_RATIO           = 0.15
MAX_PER_SOURCE_RATIO = 0.20   # nessuna sorgente > 20% del totale


# ── Utility ───────────────────────────────────────────────────────────────────

def estrai_sorgente(path: str) -> str:
    """Estrae il nome store dal path (es. 'store1', 'store8')."""
    for part in Path(path).parts:
        if re.match(r"store\d+", part, re.IGNORECASE):
            return part
    return "unknown"


def carica_coppie_valide(sharp_list_path: str) -> dict[str, list[tuple[Path, Path]]]:
    """
    Legge la lista di immagini nitide e mantiene SOLO quelle
    che hanno un label .txt non vuoto nella cartella labels/ parallela.

    Struttura attesa:
        store_X/
        ├── images/  nome.jpg   ← img_path
        └── labels/  nome.txt   ← lbl_path (cercato qui)

    Restituisce dict: { nome_store → [(img_path, lbl_path), ...] }
    """
    gruppi: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    scartate_no_label = 0
    scartate_vuoto    = 0

    with open(sharp_list_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        img_path = Path(line)
        store    = estrai_sorgente(line)

        # labels/ è sorella di images/, quindi parent di images/ è lo store
        lbl_path = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name

        if not lbl_path.exists():
            scartate_no_label += 1
            continue
        if lbl_path.stat().st_size == 0:
            scartate_vuoto += 1
            continue

        gruppi[store].append((img_path, lbl_path))

    print(f"    Senza label corrispondente : {scartate_no_label:,} scartate")
    print(f"    Con label vuoto            : {scartate_vuoto:,} scartate")
    print(f"    Coppie immagine+label OK   : {sum(len(v) for v in gruppi.values()):,}")

    return dict(gruppi)


# ── Campionamento stratificato ────────────────────────────────────────────────

def campiona_stratificato(gruppi: dict[str, list],
                          budget: int,
                          max_per_src: int,
                          seed: int) -> list:
    """
    Campiona `budget` coppie (img, lbl) distribuendo proporzionalmente
    tra le sorgenti, con un cap per sorgente.
    """
    random.seed(seed)
    totale = sum(len(v) for v in gruppi.values())
    budget = min(budget, totale)

    # Quota proporzionale con cap
    quote: dict[str, int] = {}
    for src, items in gruppi.items():
        quota_prop = round(budget * len(items) / totale)
        quote[src] = min(quota_prop, max_per_src, len(items))

    # Redistribuisci budget residuo
    usato = sum(quote.values())
    resto = budget - usato
    if resto > 0:
        candidati = sorted(gruppi.items(), key=lambda x: -(len(x[1]) - quote[x[0]]))
        for src, items in candidati:
            if resto <= 0:
                break
            aggiunta = min(resto, len(items) - quote[src], max_per_src - quote[src])
            if aggiunta > 0:
                quote[src] += aggiunta
                resto -= aggiunta

    selezionati = []
    print()
    for src in sorted(gruppi):
        items    = gruppi[src]
        n        = quote.get(src, 0)
        campione = random.sample(items, min(n, len(items)))
        selezionati.extend(campione)
        print(f"    {src:20s}: {len(items):5,} disponibili → {len(campione):4,} selezionate")

    return selezionati


def split_lista(items: list, seed: int):
    """Divide in train/val/test."""
    random.seed(seed)
    shuffled = items.copy()
    random.shuffle(shuffled)
    n       = len(shuffled)
    n_train = math.floor(n * TRAIN_RATIO)
    n_val   = math.floor(n * VAL_RATIO)
    return shuffled[:n_train], shuffled[n_train:n_train+n_val], shuffled[n_train+n_val:]


# ── Copia immagini e label ────────────────────────────────────────────────────

def copia_split(items: list[tuple[Path, Path]],
                img_dir: Path,
                lbl_dir: Path,
                split_name: str):
    """Copia coppie (immagine, label) nelle cartelle di destinazione."""
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    nomi_usati: dict[str, int] = {}

    for img_src, lbl_src in tqdm(items, desc=f"  Copia {split_name:5s}", ncols=80, leave=False):
        sorgente = estrai_sorgente(str(img_src))
        new_stem = f"{sorgente}__{img_src.stem}"
        new_name = new_stem + img_src.suffix

        # Gestisci collisioni residue
        if new_name in nomi_usati:
            nomi_usati[new_name] += 1
            new_name = f"{new_stem}_{nomi_usati[new_name]}{img_src.suffix}"
        else:
            nomi_usati[new_name] = 0

        shutil.copy2(img_src, img_dir / new_name)
        shutil.copy2(lbl_src, lbl_dir / (Path(new_name).stem + ".txt"))


# ── dataset.yaml ──────────────────────────────────────────────────────────────

def scrivi_yaml(oos_dir: Path, n_train: int, n_val: int, n_test: int):
    content = f"""# dataset.yaml — OOS Ridotto
# Generato da prepara_oos_ridotto.py
# Dataset separato da data/processed/ — non modificare quello!

path:  {oos_dir.as_posix()}

train: images/train   # {n_train} immagini
val:   images/val     # {n_val} immagini
test:  images/test    # {n_test} immagini

nc: 1
names:
  0: empty_shelf

# Note
# - Solo immagini store OOS con label YOLO non vuoti
# - Video privati esclusi (nessuna annotation disponibile)
# - Filtro blur: soglia_store=300 (varianza Laplaciano)
# - Campionamento stratificato per store, cap {MAX_PER_SOURCE_RATIO*100:.0f}% per fonte
"""
    (oos_dir / "dataset.yaml").write_text(content, encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepara data/oos_ridotto/ con sole immagini annotate"
    )
    parser.add_argument("--store_list",   required=True,
                        help="Path a sharp_paths_Store_OOS.txt")
    parser.add_argument("--target",       type=int, default=5000,
                        help="Totale immagini (default: 5000)")
    parser.add_argument("--project_root", type=str, default=".",
                        help="Cartella radice del progetto (contiene data/, src/)")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    root    = Path(args.project_root)
    oos_dir = root / "data" / "oos_ridotto"

    max_per_src = int(args.target * MAX_PER_SOURCE_RATIO)

    print(f"\n{'='*60}")
    print(f"  PREPARA DATASET OOS RIDOTTO (solo immagini annotate)")
    print(f"{'='*60}")
    print(f"  Destinazione  : {oos_dir}")
    print(f"  Target        : {args.target:,} immagini totali")
    print(f"  Cap/sorgente  : {max_per_src:,}  ({MAX_PER_SOURCE_RATIO*100:.0f}% target)")
    print(f"  Split         : train {TRAIN_RATIO*100:.0f}% / val {VAL_RATIO*100:.0f}% / test {TEST_RATIO*100:.0f}%")

    # ── Carica solo coppie con label valida ───────────────────────────────────
    print(f"\n── Filtraggio immagini (solo quelle con label) ────────────────")
    gruppi = carica_coppie_valide(args.store_list)

    totale_disponibile = sum(len(v) for v in gruppi.values())
    if totale_disponibile == 0:
        print("\n❌ Nessuna immagine con label trovata.")
        print("   Verifica che i path in store_list seguano la struttura:")
        print("   store_X/images/nome.jpg  con  store_X/labels/nome.txt parallela.")
        return

    # Avvisa se il totale disponibile è molto sotto il target
    if totale_disponibile < args.target:
        print(f"\n  ⚠️  Disponibili solo {totale_disponibile:,} immagini annotate, "
              f"meno del target {args.target:,}.")
        print(f"     Verrà usato il massimo disponibile.")

    # ── Campionamento ─────────────────────────────────────────────────────────
    print(f"\n── Campionamento stratificato ─────────────────────────────────")
    selezionati = campiona_stratificato(gruppi, args.target, max_per_src, args.seed)
    train, val, test = split_lista(selezionati, seed=args.seed)

    print(f"\n  Totale selezionate : {len(selezionati):,}")
    print(f"  train              : {len(train):,}")
    print(f"  val                : {len(val):,}")
    print(f"  test               : {len(test):,}")

    # ── Copia ─────────────────────────────────────────────────────────────────
    print(f"\n── Copia in {oos_dir} ─────────────────────────────")
    for split_name, items in [("train", train), ("val", val), ("test", test)]:
        copia_split(
            items,
            img_dir=oos_dir / "images" / split_name,
            lbl_dir=oos_dir / "labels" / split_name,
            split_name=split_name,
        )

    scrivi_yaml(oos_dir, len(train), len(val), len(test))

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ✅ COMPLETATO")
    print(f"{'='*60}")
    print(f"  data/processed/   → INVARIATO")
    print(f"  data/oos_ridotto/ → {len(selezionati):,} immagini annotate")
    print(f"    images/train/   : {len(train):,}")
    print(f"    images/val/     : {len(val):,}")
    print(f"    images/test/    : {len(test):,}")
    print(f"    labels/*/       : file .txt YOLO con annotazioni reali")
    print(f"    dataset.yaml    : pronto")
    print()
    print(f"  Nel training:")
    print(f"    build_dataloaders(data_dir='data/oos_ridotto')")


if __name__ == "__main__":
    main()