"""
scripts/train_smoke.py
=======================
Sanity check della pipeline sul dataset ridotto.

Uso:
  python scripts/train_smoke.py --data_dir data/oos_ridotto
  python scripts/train_smoke.py --data_dir data/oos_ridotto --epochs 5 --batch 2
"""

import sys, argparse, torch, torch.nn as nn, random as _random
from pathlib import Path
from tqdm import tqdm
from PIL import Image as _Image
sys.path.insert(0, ".")

from src.models.joint_pipeline import JointPipeline
from src.training.losses       import SigLIPBCELoss, YOLOLossWrapper, JointLoss
from src.data.dataset          import ShelfDataset, collate_fn
from torch.utils.data          import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",     type=str,   default="data/oos_ridotto")
    p.add_argument("--yolo_weights", type=str,   default="weights/cluster_training/phase1_yolo/weights/best.pt")
    p.add_argument("--siglip_model", type=str,   default="google/siglip2-base-patch16-224")
    p.add_argument("--roi_size",     type=int,   default=224)
    p.add_argument("--max_samples",  type=int,   default=32)
    p.add_argument("--epochs",       type=int,   default=2)
    p.add_argument("--batch",        type=int,   default=2)
    p.add_argument("--conf",         type=float, default=0.1)
    return p.parse_args()


class SmallShelfDataset(ShelfDataset):
    def __init__(self, split, data_dir="data/oos_ridotto",
                 img_size=640, augment=False, max_samples=None):
        self.img_size = img_size
        self.augment  = augment
        base    = Path(data_dir)
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        if not img_dir.exists():
            raise RuntimeError(f"Directory non trovata: {img_dir}")
        self.samples = []
        for ext in ("*.jpg", "*.png"):
            for img_path in sorted(img_dir.glob(ext)):
                lbl_path = lbl_dir / img_path.with_suffix(".txt").name
                if lbl_path.exists():
                    self.samples.append((img_path, lbl_path))
        if not self.samples:
            raise RuntimeError(f"Nessuna immagine trovata in {img_dir}.")

        # Filtra immagini corrotte
        valid = []
        for s in self.samples:
            try:
                _Image.open(s[0]).verify()
                valid.append(s)
            except Exception:
                print(f"  ⚠ Immagine corrotta ignorata: {s[0].name}")
        self.samples = valid

        if max_samples and max_samples < len(self.samples):
            _random.seed(42)
            self.samples = _random.sample(self.samples, max_samples)
        print(f"  SmallShelfDataset [{split}]: {len(self.samples)} campioni")


def run_smoke_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SMOKE TEST — device: {device}")
    print(f"  dataset : {args.data_dir}  (max {args.max_samples}/split)")
    print(f"  siglip  : {args.siglip_model}  roi={args.roi_size}px")
    print(f"  epoche  : {args.epochs}    batch: {args.batch}")
    print(f"{'='*60}\n")

    print("[1/4] Carico dataset...")
    train_ds = SmallShelfDataset("train", data_dir=args.data_dir,
                                  max_samples=args.max_samples, augment=True)
    val_ds   = SmallShelfDataset("val",   data_dir=args.data_dir,
                                  max_samples=max(args.max_samples // 4, 4))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)
    print(f"  train: {len(train_loader)} batch | val: {len(val_loader)} batch")

    print("\n[2/4] Carico pipeline...")
    pipeline = JointPipeline(
        yolo_weights      = args.yolo_weights,
        siglip_model_name = args.siglip_model,
        conf_threshold    = args.conf,
        lora_r_visual     = 4,
        lora_alpha_visual = 8,
        lora_dropout      = 0.15,
        mlp_hidden        = [256, 64],
        roi_size          = args.roi_size,
        roi_padding       = 0.15,
    ).to(device)

    print("\n[3/4] Configuro loss e ottimizzatori...")
    siglip_loss_fn = SigLIPBCELoss()
    yolo_loss_fn = YOLOLossWrapper(total_epochs=args.epochs)
    joint_loss = JointLoss(alpha=0.89, beta=0.11)
    optimizer_yolo = torch.optim.AdamW(
        list(pipeline.detector.parameters()) + list(pipeline.feature_proj.parameters()),
        lr=5e-6, weight_decay=1e-4)
    optimizer_siglip = torch.optim.AdamW(
        list(pipeline.siglip.parameters()), lr=1e-5, weight_decay=1e-2)

    print(f"\n[4/4] Training ({args.epochs} epoche)...\n")
    history = {"train_loss": [], "val_loss": [], "n_rois": []}

    for epoch in range(args.epochs):
        # ── Train ──────────────────────────────────────────────────────────────
        pipeline.train()

        t_loss = 0.0; n_rois_ep = 0; nb = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [train]",
                    ncols=100, leave=False)
        for batch in pbar:
            images = batch["images"].to(device)
            boxes  = batch["boxes"]
            labels = batch["labels"]
            optimizer_yolo.zero_grad()
            optimizer_siglip.zero_grad()

            output = pipeline(images, gt_boxes=boxes)

            loss_yolo = yolo_loss_fn.compute(
                gt_boxes    = boxes,
                gt_labels   = labels,
                images      = images,
                device      = device,
                predictions = output["predictions"],
            )
            loss_sig = (siglip_loss_fn(output["logits"], output["roi_labels_gt"])
                        if output["n_rois"] > 0
                        else torch.tensor(0.0, device=device, requires_grad=True))

            loss_j, _, _ = joint_loss(loss_yolo, loss_sig)
            loss_j.backward()
            nn.utils.clip_grad_norm_(pipeline.parameters(), max_norm=10.0)
            optimizer_yolo.step()
            optimizer_siglip.step()

            t_loss += loss_j.item(); n_rois_ep += output["n_rois"]; nb += 1
            pbar.set_postfix({"L":  f"{loss_j.item():.3f}",
                              "Ly": f"{loss_yolo.item():.3f}",
                              "Ls": f"{loss_sig.item():.3f}",
                              "ROIs": output["n_rois"]})
        t_loss /= max(nb, 1)

        # ── Val ────────────────────────────────────────────────────────────────
        pipeline.eval()
        v_loss = 0.0; nv = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                boxes  = batch["boxes"]; labels = batch["labels"]
                output = pipeline(images)
                loss_yolo = yolo_loss_fn.compute(
                    gt_boxes    = boxes,
                    gt_labels   = labels,
                    images      = images,
                    device      = device,
                    predictions = output["predictions"],
                )
                loss_sig = (siglip_loss_fn(output["logits"], output["roi_labels_gt"])
                            if output["n_rois"] > 0
                            else torch.tensor(0.0, device=device))
                v_loss += (0.89 * loss_yolo + 0.11 * loss_sig).item(); nv += 1
        v_loss /= max(nv, 1)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["n_rois"].append(n_rois_ep)
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"train={t_loss:.4f}  val={v_loss:.4f} | ROIs={n_rois_ep}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    smoke_dir = Path("weights/smoke_test")
    smoke_dir.mkdir(parents=True, exist_ok=True)
    ck_path = smoke_dir / "smoke_checkpoint.pt"
    torch.save({"epoch": args.epochs-1,
                "pipeline_state_dict": pipeline.state_dict(),
                "history": history}, ck_path)
    print(f"\n  ✓ Checkpoint salvato → {ck_path}")

    print(f"\n{'='*60}")
    print("  SMOKE TEST COMPLETATO")
    print(f"  Train loss : {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}")
    print(f"  Val loss   : {history['val_loss'][0]:.4f} → {history['val_loss'][-1]:.4f}")
    print(f"  ROIs medi  : {sum(history['n_rois'])/len(history['n_rois']):.1f}/epoca")
    if history["n_rois"][-1] == 0:
        print("\n  ⚠ Nessuna ROI — prova --conf 0.05")
    else:
        print("\n  ✓ OK — puoi procedere sul cluster con PROFILE='cluster'")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_smoke_test(parse_args())