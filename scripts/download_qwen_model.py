"""
download_qwen_model.py
=======================
Script per scaricare il modello Qwen2.5-VL sul container universitario.

Esegui UNA SOLA VOLTA dal terminale SSH (porta 2588):
  ssh -p 2588 utente@172.16.174.236
  cd /percorso/progetto
  python download_qwen_model.py

Il modello viene salvato in ./hf_models/ e riusato automaticamente
da train_phase2_qwen.py se passi --model_path.

Se il container ha accesso diretto a HuggingFace, puoi anche non
scaricare nulla — il modello viene scaricato automaticamente nel
primo training. Ma è meglio pre-scaricarlo per evitare timeout
durante i job lunghi.
"""

import os
import sys
from pathlib import Path


def main():
    # ── Verifica connessione HuggingFace ─────────────────────────────────────
    try:
        import requests
        r = requests.get("https://huggingface.co", timeout=5)
        has_internet = r.status_code == 200
    except Exception:
        has_internet = False

    if not has_internet:
        print("✗ Il container non ha accesso a HuggingFace.")
        print("  Scarica il modello esternamente e copialo con scp:")
        print()
        print("  # Dal tuo PC locale:")
        print("  pip install huggingface_hub")
        print("  python3 -c \"from huggingface_hub import snapshot_download; "
              "snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', "
              "local_dir='./Qwen2.5-VL-3B-Instruct')\"")
        print()
        print("  # Copia sul container (da PC locale):")
        print("  scp -P 2588 -r ./Qwen2.5-VL-3B-Instruct \\")
        print("      utente@172.16.174.236:/percorso/progetto/hf_models/")
        sys.exit(1)

    # ── Scarica il modello ────────────────────────────────────────────────────
    from huggingface_hub import snapshot_download

    # Scegli il modello da scaricare
    # 3B: ~6 GB su disco in bfloat16 — CONSIGLIATO
    # 7B: ~14 GB su disco — se vuoi più capacità
    MODEL_ID  = "Qwen/Qwen2.5-VL-3B-Instruct"
    LOCAL_DIR = Path("./hf_models/Qwen2.5-VL-3B-Instruct")

    # Per il 7B:
    # MODEL_ID  = "Qwen/Qwen2.5-VL-7B-Instruct"
    # LOCAL_DIR = Path("./hf_models/Qwen2.5-VL-7B-Instruct")

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Download: {MODEL_ID}")
    print(f"Destinazione: {LOCAL_DIR.resolve()}")
    print(f"Spazio necessario: ~14 GB (float32) o ~7 GB (bfloat16 safetensors)")
    print("(può richiedere 10-30 minuti in base alla banda del container)")
    print()

    snapshot_download(
        repo_id   = MODEL_ID,
        local_dir = str(LOCAL_DIR),
        # Mostra la progress bar
        tqdm_class = None,
    )

    print()
    print(f"✓ Modello scaricato in: {LOCAL_DIR.resolve()}")
    print()
    print("Usa il percorso locale nel training:")
    print(f"  python src/training/train_phase2_qwen.py \\")
    print(f"      --model_path {LOCAL_DIR}")
    print()
    print("Oppure aggiorna direttamente config_qwen.py:")
    print(f"  model_name = \"{LOCAL_DIR}\"")


if __name__ == "__main__":
    main()
