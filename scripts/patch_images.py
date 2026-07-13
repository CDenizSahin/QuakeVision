"""
QuakeVision - Gorsel Parcalama (Patching)
==========================================
xBD 1024x1024 gorselleri 512x512 patch'lere boler.
Poligon koordinatlarini yeni boyuta gore hesaplar.

Kullanim:
    python scripts/patch_images.py --images-dir data/raw/images --labels-dir data/raw/labels
"""

from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_PATCH_SIZE = 512


def load_image(path: Path) -> np.ndarray | None:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Okunamadi: {path}")
        return img
    except Exception as e:
        print(f"  [!] Gorsel hatasi: {path} -> {e}", file=sys.stderr)
        return None


def load_label_json(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [!] Etiket hatasi: {path} -> {e}", file=sys.stderr)
        return None


def parse_wkt_polygon(wkt: str) -> list[list[float]] | None:
    try:
        inner = wkt.replace("POLYGON", "").replace("((", "").replace("))", "").strip()
        pairs = inner.split(",")
        coords = []
        for pair in pairs:
            parts = pair.strip().split()
            if len(parts) >= 2:
                coords.append([float(parts[0]), float(parts[1])])
        return coords if len(coords) >= 3 else None
    except (ValueError, IndexError):
        return None


def coords_to_wkt(coords: list[list[float]]) -> str:
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    pairs_str = ", ".join(f"{c[0]:.2f} {c[1]:.2f}" for c in coords)
    return f"POLYGON (({pairs_str}))"


def clip_polygon(coords, row_off, col_off, ps):
    shifted = [[p[0] - col_off, p[1] - row_off] for p in coords]
    xs = [p[0] for p in shifted]
    ys = [p[1] for p in shifted]
    if max(xs) <= 0 or min(xs) >= ps or max(ys) <= 0 or min(ys) >= ps:
        return None
    clamped = [[max(0, min(p[0], ps)), max(0, min(p[1], ps))] for p in shifted]
    unique = set((round(p[0], 2), round(p[1], 2)) for p in clamped)
    return clamped if len(unique) >= 3 else None


def remap_label(label_data, row_off, col_off, ps):
    out = copy.deepcopy(label_data)
    if "features" not in out:
        return out
    for key in ("xy", "lng_lat"):
        feats = out["features"].get(key, [])
        if not feats:
            continue
        new_feats = []
        for feat in feats:
            wkt = feat.get("wkt", "")
            if "POLYGON" not in wkt.upper():
                continue
            coords = parse_wkt_polygon(wkt)
            if coords is None:
                continue
            new_c = clip_polygon(coords, row_off, col_off, ps)
            if new_c is None:
                continue
            nf = copy.deepcopy(feat)
            nf["wkt"] = coords_to_wkt(new_c)
            new_feats.append(nf)
        out["features"][key] = new_feats
    return out


def patch_single(img_path, lbl_path, out_img, out_lbl, ps=512):
    img = load_image(img_path)
    if img is None:
        return 0
    h, w = img.shape[:2]
    if h < ps or w < ps:
        return 0
    label_data = load_label_json(lbl_path) if lbl_path and lbl_path.exists() else None
    stem = img_path.stem
    count = 0
    for r in range(h // ps):
        for c in range(w // ps):
            ro, co = r * ps, c * ps
            patch = img[ro:ro+ps, co:co+ps].copy()
            tag = f"r{r}c{c}"
            name = f"{stem}_{tag}"
            cv2.imwrite(str(out_img / f"{name}.png"), patch)
            if label_data is not None:
                pl = remap_label(label_data, ro, co, ps)
                with open(out_lbl / f"{name}.json", "w", encoding="utf-8") as f:
                    json.dump(pl, f, indent=2)
            count += 1
    return count


def run_patching(images_dir, labels_dir, output_dir, ps=512):
    oi = output_dir / "images"
    ol = output_dir / "labels"
    oi.mkdir(parents=True, exist_ok=True)
    ol.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in exts)
    if not files:
        print(f"  [!] Gorsel bulunamadi: {images_dir}")
        return
    print(f"\n  Toplam gorsel: {len(files)}, Patch: {ps}x{ps}\n")
    total = 0
    for fp in tqdm(files, desc="  Patching", unit="img"):
        lp = labels_dir / f"{fp.stem}.json"
        total += patch_single(fp, lp if lp.exists() else None, oi, ol, ps)
    print(f"\n  Toplam {total} patch olusturuldu.")


def main():
    parser = argparse.ArgumentParser(description="xBD gorsel parcalama")
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/images"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/raw/labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/patched"))
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    args = parser.parse_args()
    print("=" * 60)
    print("  QuakeVision - Gorsel Parcalama (Patching)")
    print("=" * 60)
    run_patching(args.images_dir, args.labels_dir, args.output_dir, args.patch_size)


if __name__ == "__main__":
    main()
