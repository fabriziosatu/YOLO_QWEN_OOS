"""
filtra_blur.py
==============
Analizza e filtra le immagini blurrate dal dataset OOS.

STEP 1 — Analisi (--mode analisi):
    Calcola gli score di tutte le immagini, stampa statistiche
    e salva l'istogramma + un CSV completo per calibrare le soglie.

STEP 2 — Filtro (--mode filtra):
    Usa le soglie calibrate per:
      - copiare le blurrate in una cartella di quarantena
      - scrivere blurry_paths.txt e sharp_paths.txt

WORKFLOW CONSIGLIATO:
    1. python filtra_blur.py --mode analisi
    2. Guarda i grafici e il CSV → decidi le soglie
    3. python filtra_blur.py --mode filtra --soglia_video 500 --soglia_store 400

USO:
    pip install opencv-python numpy matplotlib tqdm
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ── Struttura dataset ─────────────────────────────────────────────────────────

BASE_VIDEO = r"C:\Users\186337\Desktop\oos1\video pepper\video_oos_pepper\Video"
BASE_OOS   = r"C:\Users\186337\Desktop\oos1\OOS dataset"

VIDEO_DIRS = {
    "video1":  os.path.join(BASE_VIDEO, "video1",  "ImmaginiOriginali"),
    "video2":  os.path.join(BASE_VIDEO, "video2",  "ImmaginiOriginali"),
    "video3":  os.path.join(BASE_VIDEO, "video3",  "ImmaginiOriginali"),
    "video4":  os.path.join(BASE_VIDEO, "video4",  "ImmaginiOriginali"),
    "video5":  os.path.join(BASE_VIDEO, "video5",  "ImmaginiOriginali"),
    "video7":  os.path.join(BASE_VIDEO, "video7",  "ImmaginiOriginali"),
    "video9":  os.path.join(BASE_VIDEO, "video9",  "ImmaginiOriginali"),
    "video10": os.path.join(BASE_VIDEO, "video10", "ImmaginiOriginali"),
    "video11": os.path.join(BASE_VIDEO, "video11", "ImmaginiOriginali"),
    "video12": os.path.join(BASE_VIDEO, "video12", "ImmaginiOriginali"),
    "video14": os.path.join(BASE_VIDEO, "video14", "ImmaginiOriginali"),
    "video15": os.path.join(BASE_VIDEO, "video15", "ImmaginiOriginali"),
    "video17": os.path.join(BASE_VIDEO, "video17", "ImmaginiOriginali"),
    "video18": os.path.join(BASE_VIDEO, "video18", "ImmaginiOriginali"),
    "video19": os.path.join(BASE_VIDEO, "video19", "ImmaginiOriginali"),
    "video20": os.path.join(BASE_VIDEO, "video20", "ImmaginiOriginali"),
    "video21": os.path.join(BASE_VIDEO, "video21", "ImmaginiOriginali"),
    "video22": os.path.join(BASE_VIDEO, "video22", "ImmaginiOriginali"),
    "video23": os.path.join(BASE_VIDEO, "video23", "ImmaginiOriginali"),
}

STORE_DIRS = {
    "store1":  os.path.join(BASE_OOS, "Grocery Products",          "store1",  "images"),
    "store2":  os.path.join(BASE_OOS, "Grocery Products",          "store2",  "images"),
    "store3":  os.path.join(BASE_OOS, "Grocery Products",          "store3",  "images"),
    "store4":  os.path.join(BASE_OOS, "Grocery Products",          "store4",  "images"),
    "store5":  os.path.join(BASE_OOS, "Grocery Products",          "store5",  "images"),
    "store6":  os.path.join(BASE_OOS, "Web Market",                "store6",  "images"),
    "store7":  os.path.join(BASE_OOS, "Supermarket1 - Agropoli",   "store7",  "images"),
    "store8":  os.path.join(BASE_OOS, "Supermarket2 - Gragnano",   "store8",  "images"),
    "store9":  os.path.join(BASE_OOS, "Supermarket3 - Salerno",    "store9",  "images"),
    "store10": os.path.join(BASE_OOS, "SKU110K",                   "store10", "images"),
    "store11_train": os.path.join(BASE_OOS, "Supermarket4",        "store11", "images", "train"),
    "store11_val":   os.path.join(BASE_OOS, "Supermarket4",        "store11", "images", "val"),
}

ESTENSIONI = ('.jpg', '.jpeg', '.png')

OUTPUT_BASE     = r"C:\Users\186337\Desktop\oos1\da_processare"
OUTPUT_BLUR_VID = os.path.join(OUTPUT_BASE, "video")
OUTPUT_BLUR_STO = os.path.join(OUTPUT_BASE, "store")


# ── Utility ───────────────────────────────────────────────────────────────────

def blur_score(path: str) -> float | None:
    """Varianza del Laplaciano su scala di grigi. Più basso = più sfocato."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def raccogli_immagini(dirs: dict) -> list[tuple[str, str]]:
    """
    Restituisce lista di (nome_sorgente, path_assoluto) per tutte le immagini
    nelle cartelle specificate (skip su cartelle mancanti con warning).
    """
    campioni = []
    for nome, dpath in dirs.items():
        if not os.path.exists(dpath):
            print(f"  ⚠️  [SKIP] Cartella non trovata: {dpath}")
            continue
        files = [f for f in os.listdir(dpath) if f.lower().endswith(ESTENSIONI)]
        for f in files:
            campioni.append((nome, os.path.join(dpath, f)))
    return campioni


# ── STEP 1: Analisi ───────────────────────────────────────────────────────────

def analisi(output_dir: str):
    """Calcola score, stampa statistiche, salva CSV e istogrammi."""
    os.makedirs(output_dir, exist_ok=True)

    for gruppo, dirs in [("Video Privati", VIDEO_DIRS), ("Store OOS", STORE_DIRS)]:
        print(f"\n{'='*55}")
        print(f"  Analisi gruppo: {gruppo}")
        print(f"{'='*55}")

        campioni = raccogli_immagini(dirs)
        print(f"  Immagini trovate: {len(campioni):,}")

        risultati = []  # (nome, path, score)
        errori    = 0

        for nome, fpath in tqdm(campioni, desc=f"  Scoring {gruppo}", ncols=80):
            score = blur_score(fpath)
            if score is None:
                errori += 1
                continue
            risultati.append((nome, fpath, score))

        if not risultati:
            print("  Nessuna immagine valida!")
            continue

        scores = [s for _, _, s in risultati]
        print(f"\n  Valide          : {len(risultati):,}")
        print(f"  Errori lettura  : {errori}")
        print(f"  Score minimo    : {min(scores):.2f}  ← più sfocata")
        print(f"  Score 10° pctile: {np.percentile(scores, 10):.2f}")
        print(f"  Score mediano   : {np.median(scores):.2f}")
        print(f"  Score medio     : {np.mean(scores):.2f}")
        print(f"  Score 90° pctile: {np.percentile(scores, 90):.2f}")
        print(f"  Score massimo   : {max(scores):.2f}  ← più nitida")

        # ── CSV ───────────────────────────────────────────────────────────────
        tag     = gruppo.replace(" ", "_")
        csv_out = os.path.join(output_dir, f"score_{tag}.csv")
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sorgente", "path", "score"])
            for nome, fpath, score in sorted(risultati, key=lambda x: x[2]):
                w.writerow([nome, fpath, f"{score:.4f}"])
        print(f"\n  ✅ CSV salvato: {csv_out}")

        # ── Istogramma ────────────────────────────────────────────────────────
        plt.figure(figsize=(13, 4))
        plt.hist(scores, bins=100, color='steelblue', edgecolor='white')
        for pct, color, label in [
            (np.percentile(scores, 10), 'red',    '10°'),
            (np.median(scores),         'orange', '50°'),
            (np.percentile(scores, 90), 'green',  '90°'),
        ]:
            plt.axvline(pct, color=color, linestyle='--', linewidth=1.5,
                        label=f'{label} pctile = {pct:.0f}')
        plt.xlabel("Varianza Laplaciano  (basso=sfocato, alto=nitido)")
        plt.ylabel("Numero immagini")
        plt.title(f"Distribuzione blur — {gruppo}  (n={len(scores):,})")
        plt.legend()
        plt.tight_layout()
        png_out = os.path.join(output_dir, f"distribuzione_{tag}.png")
        plt.savefig(png_out, dpi=150)
        plt.close()
        print(f"  ✅ Grafico salvato: {png_out}")
        print(f"\n  👉 Scegli la soglia guardando il grafico e il CSV,")
        print(f"     poi lancia: python filtra_blur.py --mode filtra \\")
        print(f"                   --soglia_video <VAL> --soglia_store <VAL>")


# ── STEP 2: Filtro ────────────────────────────────────────────────────────────

def filtra(soglia_video: float, soglia_store: float, output_dir: str):
    """
    Copia le immagini blurrate in quarantena.
    Scrive blurry_paths.txt e sharp_paths.txt per entrambi i gruppi.
    """
    os.makedirs(output_dir, exist_ok=True)

    gruppi = [
        ("Video Privati", VIDEO_DIRS, soglia_video, OUTPUT_BLUR_VID),
        ("Store OOS",     STORE_DIRS, soglia_store, OUTPUT_BLUR_STO),
    ]

    for gruppo, dirs, soglia, out_blur in gruppi:
        print(f"\n{'='*55}")
        print(f"  Filtro: {gruppo}  (soglia={soglia})")
        print(f"{'='*55}")
        os.makedirs(out_blur, exist_ok=True)

        campioni = raccogli_immagini(dirs)
        blurry_paths = []
        sharp_paths  = []
        errori       = 0

        for nome, fpath in tqdm(campioni, desc=f"  Filtrando {gruppo}", ncols=80):
            score = blur_score(fpath)
            if score is None:
                errori += 1
                continue

            if score < soglia:
                blurry_paths.append(fpath)
                # ✅ Nome univoco: sorgente + nome originale (evita collisioni)
                fname     = os.path.basename(fpath)
                dest_name = f"{nome}__{fname}"
                dest      = os.path.join(out_blur, dest_name)
                # Gestisci collisioni residue
                if os.path.exists(dest):
                    stem, ext = os.path.splitext(dest_name)
                    i = 1
                    while os.path.exists(os.path.join(out_blur, f"{stem}_{i}{ext}")):
                        i += 1
                    dest = os.path.join(out_blur, f"{stem}_{i}{ext}")
                shutil.copy2(fpath, dest)
            else:
                sharp_paths.append(fpath)

        # ── Scrivi liste di path ──────────────────────────────────────────────
        tag = gruppo.replace(" ", "_")

        blurry_txt = os.path.join(output_dir, f"blurry_paths_{tag}.txt")
        with open(blurry_txt, "w", encoding="utf-8") as f:
            f.writelines(p + "\n" for p in blurry_paths)

        sharp_txt = os.path.join(output_dir, f"sharp_paths_{tag}.txt")
        with open(sharp_txt, "w", encoding="utf-8") as f:
            f.writelines(p + "\n" for p in sharp_paths)

        totale = len(blurry_paths) + len(sharp_paths)
        print(f"\n  Totale analizzate : {totale:,}")
        print(f"  Nitide ✓          : {len(sharp_paths):,}")
        print(f"  Blurrate ✗        : {len(blurry_paths):,}  ({len(blurry_paths)/max(totale,1)*100:.1f}%)")
        print(f"  Errori lettura    : {errori}")
        print(f"  Blurrate copiate  → {out_blur}")
        print(f"  Lista blurrate    → {blurry_txt}")
        print(f"  Lista nitide      → {sharp_txt}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Blur detection — dataset OOS")
    parser.add_argument("--mode", choices=["analisi", "filtra"], required=True,
                        help="'analisi' per vedere la distribuzione, 'filtra' per applicare le soglie")
    parser.add_argument("--soglia_video", type=float, default=500,
                        help="Soglia blur per i video (default: 500)")
    parser.add_argument("--soglia_store", type=float, default=400,
                        help="Soglia blur per gli store (default: 400)")
    parser.add_argument("--output_dir", type=str,
                        default=r"C:\Users\186337\Desktop\oos1\blur_analysis",
                        help="Cartella per CSV, grafici e liste di path")
    args = parser.parse_args()

    if args.mode == "analisi":
        analisi(args.output_dir)
    else:
        filtra(args.soglia_video, args.soglia_store, args.output_dir)


if __name__ == "__main__":
    main()