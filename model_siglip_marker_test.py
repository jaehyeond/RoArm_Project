"""
model_siglip_marker_test.py
B1 VLA Foundation Model Scientist — SigLIP Marker Recognition Test

PURPOSE:
Test whether SmolVLA's frozen SigLIP encoder responds equivalently to:
  (a) Digitally-rendered visual markers (circle, arrow, crosshair)
  (b) Physically-placed visual markers (colored tape, laser dot, sticker)
     — captured through the robot's actual camera

This is the GATE test for the "physical visual prompting" research direction.

USAGE:
1. No physical markers: python model_siglip_marker_test.py --mode digital_only
2. With physical photos: python model_siglip_marker_test.py --mode full --photo_dir ./marker_photos/
3. SigLIP feature dump only: python model_siglip_marker_test.py --mode dump

INTERPRETATION:
- cosine_similarity > 0.98: modalities are equivalent from SigLIP POV → physical viable
- cosine_similarity 0.90-0.98: partial equivalence → physical usable but may need fine-tune
- cosine_similarity < 0.90: significant difference → physical prompting will NOT transfer from digital training

Created: 2026-03-25
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
import argparse
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIGLIP_MODEL_ID = "google/siglip-base-patch16-512"
# Note: SmolVLA uses a custom SigLIP variant. We use the public one here as a proxy.
# For exact SmolVLA results, extract from lerobot/lerobot/policies/smolvla/modeling_smolvla.py
# and use the vision encoder directly. This script uses the public SigLIP as approximation.

IMAGE_SIZE = 512  # SmolVLA processes images at 512x512
WORKSPACE_RESOLUTION = (720, 1280)  # Azure Kinect RGB capture size


# ---------------------------------------------------------------------------
# Marker rendering functions
# ---------------------------------------------------------------------------
def add_circle(img: Image.Image, cx: int, cy: int, radius: int = 25,
               color: str = "red", width: int = 3) -> Image.Image:
    """Add a digital circle overlay to an image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                 outline=color, width=width)
    return out


def add_arrow(img: Image.Image, cx: int, cy: int, length: int = 40,
              color: str = "red", width: int = 3) -> Image.Image:
    """Add a downward-pointing arrow to an image (like AimBot reticle)."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    # Shaft
    draw.line([(cx, cy - length), (cx, cy)], fill=color, width=width)
    # Arrowhead
    draw.polygon([(cx - 8, cy - 15), (cx + 8, cy - 15), (cx, cy)], fill=color)
    return out


def add_crosshair(img: Image.Image, cx: int, cy: int, size: int = 30,
                  color: str = "red", width: int = 2) -> Image.Image:
    """Add a crosshair (like AimBot scope reticle) to an image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.line([(cx - size, cy), (cx + size, cy)], fill=color, width=width)
    draw.line([(cx, cy - size), (cx, cy + size)], fill=color, width=width)
    draw.ellipse([cx - size // 2, cy - size // 2,
                  cx + size // 2, cy + size // 2], outline=color, width=width)
    return out


def add_trace(img: Image.Image, points: list, color: str = "blue",
              width: int = 3) -> Image.Image:
    """Add a trajectory trace (like TraceVLA) to an image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    if len(points) >= 2:
        draw.line(points, fill=color, width=width)
        # Draw endpoint dot
        last = points[-1]
        draw.ellipse([last[0] - 5, last[1] - 5, last[0] + 5, last[1] + 5],
                     fill=color)
    return out


# ---------------------------------------------------------------------------
# SigLIP embedding extraction
# ---------------------------------------------------------------------------
def load_siglip():
    """Load SigLIP model and processor."""
    try:
        from transformers import AutoProcessor, AutoModel
        print(f"Loading SigLIP: {SIGLIP_MODEL_ID}")
        processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_ID)
        model = AutoModel.from_pretrained(SIGLIP_MODEL_ID, torch_dtype=torch.float16)
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        print("SigLIP loaded.")
        return processor, model
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        raise


def get_embedding(img: Image.Image, processor, model) -> torch.Tensor:
    """Extract normalized SigLIP image embedding."""
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    inputs = processor(images=img_resized, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        # Get image features (before projection)
        outputs = model.get_image_features(**{
            k: v for k, v in inputs.items()
            if k in ['pixel_values']
        })
    emb = outputs[0].float()
    return emb / emb.norm()


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(torch.dot(a.flatten(), b.flatten()).item())


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
def test_digital_only(processor, model):
    """
    Test 1: Digital markers only.
    Baseline: clean image vs each marker type.
    Shows how much each marker perturbs SigLIP features.
    """
    print("\n" + "="*60)
    print("TEST 1: Digital marker perturbation of SigLIP features")
    print("="*60)

    # Create a synthetic workspace image (gray background, simulated objects)
    base_img = Image.new("RGB", WORKSPACE_RESOLUTION, color=(128, 100, 80))
    draw = ImageDraw.Draw(base_img)
    # Add some mock workspace objects
    draw.rectangle([200, 300, 400, 450], fill=(200, 150, 100))  # table surface
    draw.ellipse([280, 330, 360, 390], fill=(220, 50, 50))  # red object

    # Target position (center of red object)
    cx, cy = 640, 360

    # Marker variants
    markers = {
        "circle": add_circle(base_img, cx, cy),
        "arrow": add_arrow(base_img, cx, cy),
        "crosshair": add_crosshair(base_img, cx, cy),
        "trace": add_trace(base_img, [(640, 200), (640, 300), (640, 360)]),
    }

    emb_clean = get_embedding(base_img, processor, model)
    print(f"\nBaseline (clean image) computed.")

    results = {}
    for name, marked_img in markers.items():
        emb_marked = get_embedding(marked_img, processor, model)
        sim = cosine_similarity(emb_clean, emb_marked)
        results[name] = sim
        print(f"  clean vs {name:12s}: cosine_sim = {sim:.4f}  delta = {1-sim:.4f}")

    # Cross-marker similarities (do different markers look similar to SigLIP?)
    print("\nCross-marker similarities:")
    marker_items = list(markers.items())
    embs = {name: get_embedding(img, processor, model) for name, img in marker_items}
    for i, (n1, e1) in enumerate(embs.items()):
        for n2, e2 in list(embs.items())[i+1:]:
            sim = cosine_similarity(e1, e2)
            print(f"  {n1:12s} vs {n2:12s}: cosine_sim = {sim:.4f}")

    return results


def test_marker_position_sensitivity(processor, model):
    """
    Test 2: Does SigLIP encode marker POSITION?
    If the model can't distinguish marker at (300, 300) vs (600, 300),
    then visual prompting carries no spatial information.
    """
    print("\n" + "="*60)
    print("TEST 2: Marker position sensitivity")
    print("="*60)

    base_img = Image.new("RGB", WORKSPACE_RESOLUTION, color=(128, 100, 80))

    positions = [(200, 200), (400, 360), (800, 500), (1100, 300)]
    embs = []
    for cx, cy in positions:
        img = add_circle(base_img, cx, cy)
        emb = get_embedding(img, processor, model)
        embs.append((cx, cy, emb))

    print("\nPosition sensitivity (same marker, different positions):")
    for i, (cx1, cy1, e1) in enumerate(embs):
        for cx2, cy2, e2 in embs[i+1:]:
            sim = cosine_similarity(e1, e2)
            dist = ((cx2-cx1)**2 + (cy2-cy1)**2)**0.5
            print(f"  ({cx1},{cy1}) vs ({cx2},{cy2}) dist={dist:.0f}px: sim={sim:.4f}")

    print("\nInterpretation:")
    print("  sim < 0.99 at different positions → SigLIP IS position-sensitive")
    print("  sim > 0.99 at different positions → SigLIP NOT position-sensitive (bad for visual prompting)")


def test_with_real_photos(photo_dir: str, processor, model):
    """
    Test 3: Compare digital overlays vs physical markers photographed.

    SETUP REQUIRED:
    1. Take photo of workspace WITHOUT marker → save as photo_dir/clean.jpg
    2. Place RED TAPE circle on workspace floor at same position → photo_dir/physical_circle.jpg
    3. Add digital circle to clean photo → auto-generated
    4. Point green laser pointer at same position → photo_dir/physical_laser.jpg
    """
    print("\n" + "="*60)
    print("TEST 3: Digital vs Physical marker equivalence")
    print("="*60)

    clean_path = os.path.join(photo_dir, "clean.jpg")
    if not os.path.exists(clean_path):
        print(f"  SKIPPED: {clean_path} not found.")
        print("  Run setup:")
        print("  1. Take photo of empty workspace → save as marker_photos/clean.jpg")
        print("  2. Place red tape at target → marker_photos/physical_tape.jpg")
        print("  3. Point laser at target → marker_photos/physical_laser.jpg")
        return

    clean_img = Image.open(clean_path).convert("RGB")
    emb_clean = get_embedding(clean_img, processor, model)

    # Auto-generate digital overlay (user must verify the position matches physical)
    # Default: center of image
    h, w = clean_img.size[1], clean_img.size[0]
    cx, cy = w // 2, h // 2
    digital_circle = add_circle(clean_img, cx, cy, radius=40, color="red", width=4)
    emb_digital = get_embedding(digital_circle, processor, model)

    print(f"\nImages loaded from: {photo_dir}")
    results = {}

    sim_clean_digital = cosine_similarity(emb_clean, emb_digital)
    print(f"  clean vs digital_circle: sim = {sim_clean_digital:.4f}")
    results["clean_vs_digital"] = sim_clean_digital

    for marker_type in ["physical_tape", "physical_laser", "physical_sticker"]:
        path = os.path.join(photo_dir, f"{marker_type}.jpg")
        if os.path.exists(path):
            phys_img = Image.open(path).convert("RGB")
            emb_phys = get_embedding(phys_img, processor, model)

            sim_phys_digital = cosine_similarity(emb_digital, emb_phys)
            sim_clean_phys = cosine_similarity(emb_clean, emb_phys)
            print(f"  digital_circle vs {marker_type}: sim = {sim_phys_digital:.4f}")
            print(f"  clean vs {marker_type}:          sim = {sim_clean_phys:.4f}")
            results[f"digital_vs_{marker_type}"] = sim_phys_digital
        else:
            print(f"  {marker_type}: not found (skip)")

    print("\n--- GATE DECISION ---")
    key = "digital_vs_physical_tape"
    if key in results:
        sim = results[key]
        if sim > 0.98:
            print(f"PASS: sim={sim:.4f} > 0.98 → Physical tape equivalent to digital overlay")
            print("      → Train on digital overlays, deploy with physical markers. ZERO EXTRA COST.")
        elif sim > 0.90:
            print(f"PARTIAL: sim={sim:.4f} → Some difference.")
            print("         → Recommend: include 10% physical-marker episodes in training.")
        else:
            print(f"FAIL: sim={sim:.4f} < 0.90 → Physical and digital are DIFFERENT to SigLIP.")
            print("      → Must train with physical markers. All-digital training will not transfer.")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SigLIP marker recognition test")
    parser.add_argument("--mode", choices=["digital_only", "full", "dump"],
                        default="digital_only")
    parser.add_argument("--photo_dir", default="./marker_photos/",
                        help="Directory with physical marker photos (for --mode full)")
    args = parser.parse_args()

    processor, model = load_siglip()

    if args.mode in ["digital_only", "full", "dump"]:
        test_digital_only(processor, model)
        test_marker_position_sensitivity(processor, model)

    if args.mode == "full":
        test_with_real_photos(args.photo_dir, processor, model)

    print("\nDone. Results above are the gate decision for physical visual prompting research.")


if __name__ == "__main__":
    main()
