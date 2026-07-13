"""
QuakeVision - xBD JSON -> YOLOv8 Instance Segmentation Format
===============================================================
xBD etiket dosyalarini (JSON/WKT Poligon) YOLOv8 segmentasyon
formatina cevirir.

YOLO Seg formati (her satir bir nesne):
    <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
    (Tum koordinatlar [0,1] araliginda normalize edilmis)

Sinif eslemesi:
    0: no-damage     1: minor-damage
    2: major-damage  3: destroyed

Kullanim:
    python scripts/json_to_yolo.py \
        --input-dir  data/patched/labels \
        --output-dir data/processed/labels/train \
        --img-size   512
"""
from __future__ import annotations
import argparse, json, sys, random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

# Sinif esleme tablosu
DAMAGE_CLASS_MAP: dict[str, int] = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

# Gorsellestirme renkleri (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),     # yesil
    1: (0, 255, 255),   # sari
    2: (0, 128, 255),   # turuncu
    3: (0, 0, 255),     # kirmizi
}

CLASS_NAMES = {0: "no-damage", 1: "minor-damage", 2: "major-damage", 3: "destroyed"}


def parse_wkt_polygon(wkt: str) -> list[list[float]] | None:
    """WKT POLYGON -> koordinat listesi."""
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


def normalize_coords(coords: list[list[float]], img_w: int, img_h: int) -> list[float]:
    """
    Koordinatlari [0,1] araligina normalize eder.
    xBD koordinatlari (x, y) = (col, row) formatinda.
    Dondurulen liste: [x1, y1, x2, y2, ..., xn, yn] (duz liste)
    """
    flat: list[float] = []
    for x, y in coords:
        nx = max(0.0, min(x / img_w, 1.0))
        ny = max(0.0, min(y / img_h, 1.0))
        flat.extend([nx, ny])
    return flat


def get_damage_class(properties: dict[str, Any]) -> int | None:
    """xBD feature properties'den hasar sinifini cikarir."""
    # post-disaster etiketlerinde 'subtype' alani hasar seviyesini icerir
    subtype = properties.get("subtype", "").lower().strip()
    if subtype in DAMAGE_CLASS_MAP:
        return DAMAGE_CLASS_MAP[subtype]
    # Bazi etiketlerde 'damage' veya '_damage' alani olabilir
    for key in ("damage", "damage_class", "type"):
        val = properties.get(key, "").lower().strip()
        if val in DAMAGE_CLASS_MAP:
            return DAMAGE_CLASS_MAP[val]
    # Eger 'un-classified' veya tanimlanamayan bir degerse atla
    return None


def convert_single_label(
    label_path: Path,
    output_path: Path,
    img_size: int,
) -> tuple[int, int]:
    """
    Tek bir xBD JSON etiket dosyasini YOLO formatina cevirir.
    Donus: (islenen_poligon, atlanan_poligon) sayilari.
    """
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [!] JSON hatasi: {label_path} -> {e}", file=sys.stderr)
        return (0, 0)

    features = data.get("features", {}).get("xy", [])
    if not features:
        # Bos etiket dosyasi olustur (gorsel var ama bina yok)
        output_path.write_text("", encoding="utf-8")
        return (0, 0)

    lines: list[str] = []
    processed = 0
    skipped = 0

    for feat in features:
        props = feat.get("properties", {})
        wkt = feat.get("wkt", "")

        # Sinif belirle
        cls_id = get_damage_class(props)
        if cls_id is None:
            skipped += 1
            continue

        # Poligon koordinatlarini coz
        coords = parse_wkt_polygon(wkt)
        if coords is None:
            skipped += 1
            continue

        # Kapali poligonlarda son noktayi atla (ilk = son)
        if len(coords) > 3 and coords[0] == coords[-1]:
            coords = coords[:-1]

        if len(coords) < 3:
            skipped += 1
            continue

        # Normalize et
        norm = normalize_coords(coords, img_size, img_size)
        norm_str = " ".join(f"{v:.6f}" for v in norm)
        lines.append(f"{cls_id} {norm_str}")
        processed += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return (processed, skipped)


def run_conversion(
    input_dir: Path,
    output_dir: Path,
    img_size: int,
) -> None:
    """Tum JSON etiketlerini YOLO formatina cevirir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"  [!] JSON dosyasi bulunamadi: {input_dir}")
        return

    print(f"\n  Toplam JSON: {len(json_files)}")
    print(f"  Gorsel boyutu: {img_size}x{img_size}")
    print(f"  Cikti: {output_dir}\n")

    total_proc, total_skip = 0, 0
    for jp in tqdm(json_files, desc="  Donusturuluyor", unit="file"):
        out_path = output_dir / f"{jp.stem}.txt"
        p, s = convert_single_label(jp, out_path, img_size)
        total_proc += p
        total_skip += s

    print(f"\n  Islenen poligon : {total_proc}")
    print(f"  Atlanan poligon : {total_skip}")


# ──────────────────────────────────────────────
# Sanity Check (Dogrulama)
# ──────────────────────────────────────────────
def sanity_check(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    img_size: int = 512,
    count: int = 5,
) -> None:
    """
    Rastgele N gorsel uzerine YOLO poligonlarini cizer
    ve sonuclari output_dir'e kaydeder.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(labels_dir.glob("*.txt"))
    # Icerik bos olmayanlari filtrele
    non_empty = [f for f in txt_files if f.stat().st_size > 5]
    if not non_empty:
        print("  [!] Dogrulama icin uygun etiket bulunamadi.")
        return

    samples = random.sample(non_empty, min(count, len(non_empty)))
    print(f"\n  Sanity Check: {len(samples)} gorsel kontrol ediliyor...\n")

    for txt_path in samples:
        stem = txt_path.stem
        # Gorseli bul
        img_path = None
        for ext in (".png", ".jpg", ".jpeg", ".tif"):
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"  [!] Gorsel bulunamadi: {stem}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        lines = txt_path.read_text(encoding="utf-8").strip().split("\n")

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:  # en az class + 3 nokta (6 deger)
                continue
            cls_id = int(parts[0])
            vals = list(map(float, parts[1:]))
            # Denormalize
            pts = []
            for i in range(0, len(vals), 2):
                px = int(vals[i] * w)
                py = int(vals[i + 1] * h)
                pts.append([px, py])
            pts_arr = np.array(pts, dtype=np.int32)
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            cv2.polylines(img, [pts_arr], isClosed=True, color=color, thickness=2)
            # Sinif etiketi
            if len(pts) > 0:
                cv2.putText(
                    img, CLASS_NAMES.get(cls_id, "?"),
                    (pts[0][0], pts[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                )

        save_path = output_dir / f"check_{stem}.png"
        cv2.imwrite(str(save_path), img)
        print(f"  [+] Kaydedildi: {save_path.name}")

    print(f"\n  Sanity check tamamlandi -> {output_dir}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="xBD JSON -> YOLOv8-Seg donusturucu")
    parser.add_argument("--input-dir", type=Path, default=Path("data/patched/labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/labels/train"))
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument(
        "--sanity-check", action="store_true",
        help="Donusumden sonra dogrulama gorselleri olustur",
    )
    parser.add_argument("--sanity-images-dir", type=Path, default=Path("data/patched/images"))
    parser.add_argument("--sanity-output-dir", type=Path, default=Path("data/sanity_check"))
    parser.add_argument("--sanity-count", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("  QuakeVision - JSON -> YOLOv8-Seg Donusturucu")
    print("=" * 60)

    run_conversion(args.input_dir, args.output_dir, args.img_size)

    if args.sanity_check:
        sanity_check(
            args.sanity_images_dir,
            args.output_dir,
            args.sanity_output_dir,
            args.img_size,
            args.sanity_count,
        )


if __name__ == "__main__":
    main()
