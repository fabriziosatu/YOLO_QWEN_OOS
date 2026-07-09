"""
src/data/dataset.py
====================
Dataset PyTorch per il caricamento di immagini e label.

Supporta due modalità operative:

  mode='phase1'  — solo positivi (empty_shelf annotati)
                   Usato per il training standalone di YOLO (Fase 1)
                   e per val/test in entrambe le fasi.

  mode='phase2'  — positivi + negativi bilanciati dinamicamente
                   Usato per il training della pipeline joint (Fase 2).
                   Ad ogni epoca viene campionato un sottoinsieme di
                   negativi pari al numero di positivi (rapporto 1:1).
                   I negativi ruotano tra le epoche.

Ogni campione restituisce sempre almeno:
  - image:        Tensor [3, H, W] normalizzato in [0,1]
  - boxes:        Tensor [N, 4] formato YOLO (xc, yc, w, h)
  - labels:       Tensor [N]   tutti 0 (empty_shelf)
  - image_path:   str
  - is_negative:  bool

In phase2 aggiunge:
  - neg_image:       Tensor [3, H, W]  immagine negativa abbinata
  - neg_boxes:       Tensor [M, 4]     box prodotti (x1n y1n x2n y2n)
  - neg_image_path:  str
"""

import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ── Costanti ──────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data/processed_clean")
IMG_SUFFIX = {".jpg", ".jpeg", ".png"}

NEG_CONFIRMED_FILE = DATA_DIR / "negatives" / "confirmed.txt"
NEG_SKU_DIR        = DATA_DIR / "negatives" / "sku110k"
NEG_WEB_DIR        = DATA_DIR / "negatives" / "webmarket"


# ── Caricamento pool negativi ─────────────────────────────────────────────────

def _load_negative_pool(sku_img_dir: Path, web_img_dir: Path) -> list:
    """
    Carica il pool completo di negativi da tutte le sorgenti.
    Ogni elemento e' un dict con:
      'img_path'  : Path all'immagine (o None se risolta lazy)
      'boxes_path': Path al .txt con box prodotto (o None)
      'source'    : 'confirmed' | 'sku110k' | 'webmarket'
    """
    pool = []

    # 1. Negativi confermati (XML vuoti da video_oos_pepper)
    if NEG_CONFIRMED_FILE.exists():
        for line in NEG_CONFIRMED_FILE.read_text().splitlines():
            p = Path(line.strip())
            if p.exists():
                pool.append({
                    "img_path":   p,
                    "boxes_path": None,
                    "source":     "confirmed",
                })

    # 2. Hard negatives SKU110K
    if NEG_SKU_DIR.exists():
        for txt_path in sorted(NEG_SKU_DIR.rglob("*.txt")):
            img_path = sku_img_dir / f"{txt_path.stem}.jpg"
            pool.append({
                "img_path":   img_path if img_path.exists() else None,
                "boxes_path": txt_path,
                "source":     "sku110k",
            })

    # 3. Hard negatives WebMarket
    if NEG_WEB_DIR.exists():
        for txt_path in sorted(NEG_WEB_DIR.glob("*.txt")):
            img_path = web_img_dir / f"{txt_path.stem}.jpg"
            pool.append({
                "img_path":   img_path if img_path.exists() else None,
                "boxes_path": txt_path,
                "source":     "webmarket",
            })

    return pool


def _load_neg_boxes(boxes_path) -> list:
    """
    Legge un file .txt di negativi.
    Formato: x1_norm y1_norm x2_norm y2_norm (una box per riga).
    """
    if boxes_path is None or not Path(boxes_path).exists():
        return []
    boxes = []
    with open(boxes_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                try:
                    boxes.append(tuple(float(p) for p in parts))
                except ValueError:
                    continue
    return boxes


# ── Dataset ───────────────────────────────────────────────────────────────────

class ShelfDataset(Dataset):
    """
    Dataset per il rilevamento di spazi vuoti su scaffali.

    Args:
        split:       'train', 'val' o 'test'
        img_size:    dimensione input quadrata (default 640)
        augment:     augmentazioni (solo train)
        mode:        'phase1' o 'phase2'
        data_dir:    cartella root dataset (default: data/processed_clean)
        sku_img_dir: cartella immagini SKU110K
        web_img_dir: cartella immagini WebMarket
    """

    def __init__(
        self,
        split,
        img_size=640,
        augment=False,
        mode="phase1",
        data_dir=DATA_DIR,
        sku_img_dir=Path("data/raw/sku110k/images"),
        web_img_dir=Path("data/raw/WebMarket/images"),
    ):
        assert mode in ("phase1", "phase2"), \
            f"mode deve essere 'phase1' o 'phase2', ricevuto: '{mode}'"

        self.img_size    = img_size
        self.augment     = augment
        self.mode        = mode
        self.sku_img_dir = Path(sku_img_dir)
        self.web_img_dir = Path(web_img_dir)

        data_dir = Path(data_dir)
        # Supporta due strutture:
        #   A) data_dir/images/<split>/  (struttura standard)
        #   B) data_dir/<split>/images/  (struttura cluster HPC)
        if (data_dir / "images" / split).exists():
            img_dir = data_dir / "images" / split
            lbl_dir = data_dir / "labels" / split
        else:
            img_dir = data_dir / split / "images"
            lbl_dir = data_dir / split / "labels"

        # Carica positivi
        self.positives = []
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in IMG_SUFFIX:
                continue
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if lbl_path.exists():
                self.positives.append((img_path, lbl_path))

        if not self.positives:
            raise RuntimeError(
                f"Nessun positivo trovato in {img_dir}. "
                f"Hai eseguito prepare_dataset.py?"
            )

        # Carica pool negativi (solo phase2 e solo train)
        self._neg_pool    = []
        self._neg_sampled = []

        if mode == "phase2" and split == "train":
            self._neg_pool = _load_negative_pool(self.sku_img_dir, self.web_img_dir)
            if not self._neg_pool:
                print("  ⚠ ShelfDataset phase2: nessun negativo trovato.")
            else:
                self._resample_negatives()

        print(f"  ShelfDataset [{split}|{mode}]: "
              f"{len(self.positives)} positivi, "
              f"{len(self._neg_sampled)} negativi per epoca")

    # ── API pubblica ──────────────────────────────────────────────────────────

    def resample_negatives(self):
        """
        Ricampiona i negativi per la nuova epoca (rapporto 1:1).
        Chiamare dal trainer all'inizio di ogni epoca:
            train_loader.dataset.resample_negatives()
        """
        if self.mode == "phase2" and self._neg_pool:
            self._resample_negatives()

    def __len__(self):
        return len(self.positives) + len(self._neg_sampled)

    def __getitem__(self, idx):
        if idx < len(self.positives):
            return self._load_positive(idx)
        else:
            return self._load_negative(idx - len(self.positives))

    # ── Metodi privati ────────────────────────────────────────────────────────

    def _resample_negatives(self):
        n = len(self.positives)
        if len(self._neg_pool) <= n:
            self._neg_sampled = list(self._neg_pool)
        else:
            self._neg_sampled = random.sample(self._neg_pool, n)

    def _load_positive(self, idx):
        img_path, lbl_path = self.positives[idx]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img, pad_info  = self._letterbox(img)

        boxes = self._load_yolo_labels(lbl_path)
        if boxes:
            boxes = self._adjust_boxes_letterbox(boxes, orig_w, orig_h, pad_info)
        if self.augment and boxes:
            img, boxes = self._augment(img, boxes)

        img_t    = TF.to_tensor(img)
        boxes_t  = torch.tensor(boxes, dtype=torch.float32) if boxes \
                   else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.zeros(len(boxes), dtype=torch.long)

        sample = {
            "image":       img_t,
            "boxes":       boxes_t,
            "labels":      labels_t,
            "image_path":  str(img_path),
            "is_negative": False,
        }

        if self.mode == "phase2":
            sample["neg_image"]      = torch.zeros_like(img_t)
            sample["neg_boxes"]      = torch.zeros((0, 4), dtype=torch.float32)
            sample["neg_image_path"] = ""

        return sample

    def _load_negative(self, idx):
        neg      = self._neg_sampled[idx]
        img_path = neg["img_path"]

        img_t = None
        if img_path is not None and Path(img_path).exists():
            try:
                from PIL import ImageFile
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                img = Image.open(img_path).convert("RGB")
                img, _ = self._letterbox(img)
                img_t  = TF.to_tensor(img)
            except Exception:
                img_t = None

        if img_t is None:
            img_t = torch.zeros(3, self.img_size, self.img_size)

        neg_boxes   = _load_neg_boxes(neg.get("boxes_path"))
        neg_boxes_t = torch.tensor(neg_boxes, dtype=torch.float32) if neg_boxes \
                      else torch.zeros((0, 4), dtype=torch.float32)

        return {
            "image":         img_t,
            "boxes":         torch.zeros((0, 4), dtype=torch.float32),
            "labels":        torch.zeros(0, dtype=torch.long),
            "image_path":    str(img_path) if img_path else "",
            "is_negative":   True,
            "neg_image":     img_t,
            "neg_boxes":     neg_boxes_t,
            "neg_image_path": str(img_path) if img_path else "",
        }

    def _load_yolo_labels(self, lbl_path):
        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        xc, yc, w, h = (float(p) for p in parts[1:])
                        if w > 0 and h > 0:
                            boxes.append((xc, yc, w, h))
                    except ValueError:
                        continue
        return boxes

    def _letterbox(self, img):
        target = self.img_size
        orig_w, orig_h = img.size
        scale    = min(target / orig_w, target / orig_h)
        new_w    = int(orig_w * scale)
        new_h    = int(orig_h * scale)
        img      = img.resize((new_w, new_h), Image.BILINEAR)
        pad_left = (target - new_w) // 2
        pad_top  = (target - new_h) // 2
        canvas   = Image.new("RGB", (target, target), (114, 114, 114))
        canvas.paste(img, (pad_left, pad_top))
        return canvas, (scale, pad_left, pad_top)

    def _adjust_boxes_letterbox(self, boxes, orig_w, orig_h, pad_info):
        scale, pad_left, pad_top = pad_info
        target   = self.img_size
        adjusted = []
        for (xc, yc, w, h) in boxes:
            xc_new = (xc * orig_w * scale + pad_left) / target
            yc_new = (yc * orig_h * scale + pad_top)  / target
            w_new  = (w  * orig_w * scale)             / target
            h_new  = (h  * orig_h * scale)             / target
            xc_new = min(max(xc_new, 0.0), 1.0)
            yc_new = min(max(yc_new, 0.0), 1.0)
            w_new  = min(max(w_new,  0.0), 1.0)
            h_new  = min(max(h_new,  0.0), 1.0)
            if w_new > 0 and h_new > 0:
                adjusted.append((xc_new, yc_new, w_new, h_new))
        return adjusted

    def _augment(self, img, boxes):
        if random.random() < 0.5:
            img   = TF.hflip(img)
            boxes = [(1.0 - xc, yc, w, h) for (xc, yc, w, h) in boxes]
        if random.random() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
        return img, boxes


# ── Collate functions ─────────────────────────────────────────────────────────

def collate_fn(batch):
    """Collate standard per phase1 — retrocompatibile con il codice esistente."""
    return {
        "images":      torch.stack([b["image"]      for b in batch]),
        "boxes":       [b["boxes"]      for b in batch],
        "labels":      [b["labels"]     for b in batch],
        "image_paths": [b["image_path"] for b in batch],
    }


def collate_fn_phase2(batch):
    """
    Collate per phase2.
    Aggiunge is_negative, neg_images e neg_boxes al batch standard.
    La JointPipeline usa is_negative per distinguere i sample
    e assegnare le label GT corrette alle ROI di SigLIP.
    """
    return {
        # Campi standard (compatibili con YOLO e analyze_losses)
        "images":      torch.stack([b["image"]  for b in batch]),
        "boxes":       [b["boxes"]  for b in batch],
        "labels":      [b["labels"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],

        # Maschera negativi
        "is_negative": torch.tensor([b["is_negative"] for b in batch]),

        # Immagini e box negative per SigLIP
        "neg_images":      torch.stack([b["neg_image"]  for b in batch]),
        "neg_boxes":       [b["neg_boxes"] for b in batch],
        "neg_image_paths": [b["neg_image_path"] for b in batch],
    }


# ── Factory function ──────────────────────────────────────────────────────────

def build_dataloaders(
    img_size    = 640,
    batch_size  = 8,
    mode        = "phase1",
    data_dir    = DATA_DIR,
    sku_img_dir = Path("data/raw/sku110k/images"),
    web_img_dir = Path("data/raw/WebMarket/images"),
):
    """
    Costruisce i DataLoader per train, val e test.

    Args:
        mode: 'phase1' — training YOLO standalone
              'phase2' — training pipeline joint con negativi

    Uso nel trainer di Fase 2 (all'inizio di ogni epoca):
        train_loader.dataset.resample_negatives()
    """
    assert mode in ("phase1", "phase2")

    common = dict(
        img_size    = img_size,
        data_dir    = data_dir,
        sku_img_dir = sku_img_dir,
        web_img_dir = web_img_dir,
    )

    # val e test usano sempre solo positivi per metriche corrette
    train_ds = ShelfDataset("train", augment=True,  mode=mode,     **common)
    val_ds   = ShelfDataset("val",   augment=False, mode="phase1", **common)
    test_ds  = ShelfDataset("test",  augment=False, mode="phase1", **common)

    _collate_train = collate_fn_phase2 if mode == "phase2" else collate_fn

    n_workers = min(4, torch.multiprocessing.cpu_count())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, collate_fn=_collate_train,
        pin_memory=True, persistent_workers=(n_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=(n_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=(n_workers > 0),
    )

    return train_loader, val_loader, test_loader


# ── Test rapido ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Test phase1...")
    tl, vl, _ = build_dataloaders(batch_size=4, mode="phase1")
    b = next(iter(tl))
    print(f"  images: {b['images'].shape}  boxes[0]: {b['boxes'][0].shape}")

    print("\nTest phase2...")
    tl2, _, _ = build_dataloaders(batch_size=4, mode="phase2")
    b2 = next(iter(tl2))
    print(f"  images:      {b2['images'].shape}")
    print(f"  is_negative: {b2['is_negative'].tolist()}")
    print(f"  neg_images:  {b2['neg_images'].shape}")

    print("\nTest resample_negatives (simula inizio nuova epoca)...")
    tl2.dataset.resample_negatives()
    print(f"  ✓ {len(tl2.dataset._neg_sampled)} negativi ricampionati")

    print("\n✓ Dataset OK!")