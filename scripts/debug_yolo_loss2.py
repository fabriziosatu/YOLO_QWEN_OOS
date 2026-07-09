"""
scripts/debug_yolo_loss2.py
============================
Ispezione approfondita della struttura interna dell'output di YOLO26
e di come model.loss si aspetta di riceverlo.

Esegui con:
    python scripts/debug_yolo_loss2.py
"""

import sys
sys.path.insert(0, ".")

import torch
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

yolo  = YOLO("yolo26n.pt")
model = yolo.model.to(device)
model.train()

dummy = torch.rand(2, 3, 640, 640).to(device)

# ── Cattura l'output del Detect layer ────────────────────────────────────────
raw_captured = {}

def hook_fn(module, input, output):
    raw_captured["output"] = output

hook = model.model[-1].register_forward_hook(hook_fn)

with torch.no_grad():
    _ = model(dummy)

hook.remove()

raw = raw_captured["output"]

# ── Ispeziona la struttura ricorsivamente ─────────────────────────────────────
def describe(obj, name="", indent=0):
    pad = "  " * indent
    if isinstance(obj, torch.Tensor):
        print(f"{pad}{name}: Tensor {tuple(obj.shape)} dtype={obj.dtype}")
    elif isinstance(obj, dict):
        print(f"{pad}{name}: dict con chiavi {list(obj.keys())}")
        for k, v in obj.items():
            describe(v, k, indent + 1)
    elif isinstance(obj, (list, tuple)):
        tname = type(obj).__name__
        print(f"{pad}{name}: {tname} di {len(obj)} elementi")
        for i, v in enumerate(obj):
            describe(v, f"[{i}]", indent + 1)
    else:
        print(f"{pad}{name}: {type(obj).__name__} = {obj}")

print("=== Struttura output Detect layer ===")
describe(raw, "raw")

# ── Prova model.loss con diverse combinazioni ─────────────────────────────────
print("\n=== Tentativi model.loss ===")

batch = {
    "img":       dummy,
    "cls":       torch.zeros(4, device=device),
    "bboxes":    torch.rand(4, 4, device=device),
    "batch_idx": torch.tensor([0,0,1,1], dtype=torch.float32, device=device),
}

# Tentativo 1: passa il dict raw direttamente
print("\nTentativo 1: model.loss(batch, preds=raw)")
try:
    loss, items = model.loss(batch, preds=raw)
    print(f"  ✓ OK — loss={loss.item():.4f}")
except Exception as e:
    print(f"  ✗ {e}")

# Tentativo 2: passa solo one2many
print("\nTentativo 2: model.loss(batch, preds=raw['one2many'])")
try:
    loss, items = model.loss(batch, preds=raw["one2many"])
    print(f"  ✓ OK — loss={loss.item():.4f}")
except Exception as e:
    print(f"  ✗ {e}")

# Tentativo 3: passa solo one2one
print("\nTentativo 3: model.loss(batch, preds=raw['one2one'])")
try:
    loss, items = model.loss(batch, preds=raw["one2one"])
    print(f"  ✓ OK — loss={loss.item():.4f}")
except Exception as e:
    print(f"  ✗ {e}")

# Tentativo 4: cattura input del Detect layer invece dell'output
print("\nTentativo 4: cattura INPUT del Detect layer...")
input_captured = {}

def hook_input(module, input, output):
    input_captured["input"] = input

hook2 = model.model[-1].register_forward_hook(hook_input)
with torch.no_grad():
    _ = model(dummy)
hook2.remove()

inp = input_captured["input"]
print(f"  Input è: {type(inp).__name__} di {len(inp)} elementi")
for i, x in enumerate(inp):
    if isinstance(x, torch.Tensor):
        print(f"    [{i}] Tensor {tuple(x.shape)}")
    elif isinstance(x, (list, tuple)):
        print(f"    [{i}] {type(x).__name__} di {len(x)}")
        for j, y in enumerate(x):
            if isinstance(y, torch.Tensor):
                print(f"      [{j}] Tensor {tuple(y.shape)}")

print("\nTentativo 5: model.loss(batch, preds=input[0])")
try:
    loss, items = model.loss(batch, preds=inp[0])
    print(f"  ✓ OK — loss={loss.item():.4f}")
except Exception as e:
    print(f"  ✗ {e}")

# Tentativo 6: usa model.loss senza preds (solo batch)
print("\nTentativo 6: model.loss(batch) senza preds — fa forward interno")
try:
    loss, items = model.loss(batch)
    print(f"  ✓ OK — loss={loss.item():.4f}")
    print("  → model.loss fa forward interno autonomamente!")
except Exception as e:
    print(f"  ✗ {e}")

print("\nFine debug.")