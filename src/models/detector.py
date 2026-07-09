"""
src/models/detector.py
=======================
Wrapper attorno a YOLO26 con predizioni completamente differenziabili.

SOLUZIONE AL PROBLEMA DEL DETACH():
  YOLO26 di default carica i parametri con requires_grad=False e
  usa detach() internamente nel Detect layer (x_detach = [xi.detach() for xi in x]).

  Soluzione in due passi:
  1. Sostituisce il Detect layer con DetectNoDetach che rimuove il detach()
     SOLO in train mode — in eval mode il comportamento e' identico all'originale
  2. Abilita requires_grad=True su tutti i parametri del modello

  Risultato: decoded boxes con grad_fn=True in pixel assoluti [0,640]
  che permettono un unico backward sulla joint loss.
  In eval mode NMS funziona correttamente come nell'originale.

DECODIFICA DIFFERENZIABILE:
  In train mode YOLO produce raw_boxes in formato DFL non decodificato.
  Usiamo decode_bboxes() + make_anchors() di Ultralytics per decodificare
  in pixel assoluti mantenendo il grafo computazionale intatto.

  Soglia confidenza per matching: 0.1
  -> durante il training vogliamo trovare le predizioni candidate
     per ogni GT box, non fare inferenza finale
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors

FEATURE_LAYER_IDX = 13
FEATURE_CHANNELS  = 128
FEATURE_STRIDE    = 16

# Soglia confidenza per il matching durante il training
TRAIN_CONF_THRESHOLD = 0.1


class DetectNoDetach(Detect):
    """
    Sottoclasse di Detect che rimuove il detach() interno SOLO in train mode.

    In train mode:
        # RIMOSSA la riga con detach():
        # x_detach = [xi.detach() for xi in x]
        one2one = self.forward_head(x, **self.one2one)  <- x diretto

    In eval mode:
        # IDENTICO all'originale — NMS funziona correttamente
        x_detach = [xi.detach() for xi in x]
        one2one = self.forward_head(x_detach, **self.one2one)

    Questo garantisce che:
    - In training: i gradienti fluiscono attraverso la testa one2one
    - In eval: il NMS riceve dati nel formato atteso da Ultralytics
    """

    def forward(self, x):
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            if self.training:
                # TRAIN MODE: rimuove detach() per grafo differenziabile
                one2one = self.forward_head(x, **self.one2one)
            else:
                # EVAL MODE: comportamento identico all'originale
                # il detach() e' necessario per il NMS di Ultralytics
                x_detach = [xi.detach() for xi in x]
                one2one  = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)


class YOLO26Detector(nn.Module):

    def __init__(self, weights: str = "yolo26n.pt", conf_threshold: float = 0.25):
        super().__init__()
        self.conf_threshold       = conf_threshold
        self.train_conf_threshold = TRAIN_CONF_THRESHOLD
        self.weights              = weights

        yolo = YOLO(weights)
        self.model = yolo.model

        # ── Passo 1: sostituisci Detect con DetectNoDetach ────────────────────
        old_detect = self.model.model[-1]
        new_detect = DetectNoDetach.__new__(DetectNoDetach)
        new_detect.__dict__.update(old_detect.__dict__)
        new_detect.__class__ = DetectNoDetach
        self.model.model[-1] = new_detect

        # ── Passo 2: abilita requires_grad su tutti i parametri ───────────────
        for p in self.model.parameters():
            p.requires_grad_(True)

        # ── Passo 3: congela il Detect layer ─────────────────────────────────
        # Preserva anchors e strides calibrati — necessari per la decodifica
        for p in self.model.model[-1].parameters():
            p.requires_grad_(False)

        self.model.train()

        # ── Hook: feature map dal layer 13 (per RoI Align) ───────────────────
        self._feature_map = None
        self._hook_feat   = self.model.model[FEATURE_LAYER_IDX].register_forward_hook(
            lambda m, i, o: setattr(self, "_feature_map", o)
        )

    def forward(self, images: torch.Tensor):
        """
        Forward pass con predizioni differenziabili.

        Train mode:
          - Cattura feature map dal layer 13 (con grad_fn)
          - Decodifica le box in pixel assoluti mantenendo grad_fn
          - Restituisce predizioni (B, N, 6) differenziabili

        Eval mode:
          - Usa il forward standard di Ultralytics con NMS corretto
          - DetectNoDetach ripristina il detach() in eval mode
          - Restituisce predizioni post-NMS accurate

        Returns:
            predictions : (B, N, 6) [x1,y1,x2,y2,conf,cls] in pixel
            feature_map : (B, 128, 40, 40) con grad_fn in train, senza in eval
            raw_detect  : dict con one2one, one2many
        """
        self._feature_map = None

        if self.training:
            # Forward in train mode — DetectNoDetach mantiene grad_fn
            raw_out     = self.model(images)
            feature_map = self._feature_map
            predictions = self._decode_train_predictions(raw_out, images.device)
            return predictions, feature_map, raw_out

        else:
            # Eval mode — NMS corretto grazie al detach() ripristinato
            with torch.no_grad():
                eval_out    = self.model(images)
            feature_map = self._feature_map

            if isinstance(eval_out, (list, tuple)):
                predictions = eval_out[0]
            else:
                predictions = torch.zeros(
                    images.shape[0], 300, 6, device=images.device
                )

            return predictions, feature_map, eval_out

    def _decode_train_predictions(
        self, raw_out: dict, device: torch.device
    ) -> torch.Tensor:
        """
        Decodifica le predizioni del train mode in pixel assoluti.

        Usa decode_bboxes() e make_anchors() di Ultralytics per
        convertire le raw_boxes in coordinate xyxy [0,640],
        mantenendo il grafo computazionale per il backward.

        Returns:
            (B, N, 6) tensor con [x1,y1,x2,y2,conf,cls] e grad_fn
        """
        detect    = self.model.model[-1]
        one2one   = raw_out["one2one"]
        raw_boxes = one2one["boxes"]    # (B, 4, 8400)
        scores    = one2one["scores"]   # (B, nc, 8400)
        feats     = one2one["feats"]    # lista di 3 feature map

        # Calcola anchors e strides dai feature level
        anchors, strides = (
            a.transpose(0, 1)
            for a in make_anchors(feats, detect.stride, 0.5)
        )

        # Decodifica: raw_boxes -> pixel assoluti (mantiene grad_fn)
        decoded = detect.decode_bboxes(
            detect.dfl(raw_boxes),
            anchors.unsqueeze(0)
        ) * strides
        # decoded shape: (B, 4, 8400) in formato xyxy [0,640]

        # Confidenza: max score tra le classi
        conf = scores.sigmoid().max(dim=1, keepdim=True).values  # (B,1,8400)

        # Classe predetta
        cls  = scores.sigmoid().argmax(dim=1, keepdim=True).float()  # (B,1,8400)

        # Assembla output (B, 8400, 6): [x1,y1,x2,y2,conf,cls]
        predictions = torch.cat([
            decoded.permute(0, 2, 1),   # (B, 8400, 4)
            conf.permute(0, 2, 1),      # (B, 8400, 1)
            cls.permute(0, 2, 1),       # (B, 8400, 1)
        ], dim=2)                        # (B, 8400, 6)

        return predictions

    def filter_predictions(self, predictions: torch.Tensor):
        """
        Filtra le predizioni per confidenza.

        In train mode usa soglia bassa (0.1) per il matching IoU.
        In eval mode usa la soglia standard (0.25).
        """
        threshold = (self.train_conf_threshold
                     if self.training else self.conf_threshold)

        filtered = []
        for pred in predictions:
            with torch.no_grad():
                mask = pred[:, 4] >= threshold
            # mantieni grad_fn sulle predizioni filtrate
            filtered.append(pred[mask])
        return filtered

    def predictions_to_roi_format(self, filtered_preds: list, batch_size: int):
        """Converte le predizioni nel formato [batch_idx, x1, y1, x2, y2]."""
        roi_list = []
        for batch_idx, preds in enumerate(filtered_preds):
            if len(preds) == 0:
                continue
            idx_col = torch.full(
                (len(preds), 1), batch_idx,
                dtype=torch.float32, device=preds.device
            )
            roi_list.append(torch.cat([idx_col, preds[:, :4]], dim=1))

        if roi_list:
            return torch.cat(roi_list, dim=0)
        device = filtered_preds[0].device if filtered_preds else torch.device("cpu")
        return torch.zeros((0, 5), dtype=torch.float32, device=device)

    def remove_hooks(self):
        self._hook_feat.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    print("Test YOLO26Detector differenziabile...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.utils.config import CFG, PHASE1_WEIGHTS
    det = YOLO26Detector(str(PHASE1_WEIGHTS), conf_threshold=0.25).to(device)

    # Test train mode
    det.train()
    from src.data.dataset import build_dataloaders
    loader, _, _ = build_dataloaders(
        img_size=640, batch_size=2,
        data_dir=str(CFG.data.data_dir), num_workers=0
    )
    batch  = next(iter(loader))
    images = batch["images"].to(device)
    boxes  = batch["boxes"]

    preds, fmap, raw = det(images)
    print(f"[TRAIN] predictions shape : {preds.shape}")
    print(f"[TRAIN] predictions grad_fn: {preds.grad_fn is not None}")
    print(f"[TRAIN] feature_map grad_fn: {fmap.grad_fn is not None}")

    filtered = det.filter_predictions(preds)
    print(f"[TRAIN] Predizioni sopra soglia {det.train_conf_threshold}:")
    for i, fp in enumerate(filtered):
        print(f"  img {i}: {len(fp)} predizioni")

    from torchvision.ops import box_iou
    gt = boxes[0].to(device)
    xc,yc,w,h = gt[:,0]*640, gt[:,1]*640, gt[:,2]*640, gt[:,3]*640
    gt_xyxy = torch.stack([xc-w/2, yc-h/2, xc+w/2, yc+h/2], dim=1)
    if len(filtered[0]) > 0:
        with torch.no_grad():
            iou = box_iou(filtered[0][:,:4], gt_xyxy)
        print(f"[TRAIN] Max IoU con GT: {iou.max().item():.4f}")

    # Test eval mode
    det.eval()
    preds_eval, fmap_eval, _ = det(images)
    print(f"\n[EVAL] predictions shape : {preds_eval.shape}")
    print(f"[EVAL] n predizioni img0 : {len(det.filter_predictions(preds_eval)[0])}")

    print("\n✓ YOLO26Detector OK!")