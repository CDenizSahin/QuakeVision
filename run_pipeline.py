"""
QuakeVision - Ana Pipeline
============================
Tum on isleme adimlarini sirasiyla calistirir:
  1. Dizin yapisi kurulumu
  2. Gorsel parcalama (patching)
  3. JSON -> YOLOv8-Seg donusumu
  4. (Opsiyonel) Sanity check

Kullanim:
    python run_pipeline.py
    python run_pipeline.py --skip-patch        # Sadece YOLO donusumu
    python run_pipeline.py --sanity-check      # Dogrulama dahil
"""

from __future__ import annotations
import argparse, shutil, random
from pathlib import Path

# Proje kokundeki modulleri iceri aktar
from setup_dirs import setup_directories
from scripts.patch_images import run_patching
from scripts.json_to_yolo import run_conversion, sanity_check

PROJECT_ROOT = Path(__file__).resolve().parent


def split_train_val(
    src_images: Path,
    src_labels: Path,
    dst_root: Path,
    val_ratio: float = 0.15,
) -> None:
    """Patch'lenmis gorselleri train/val olarak boler."""
    print("\n" + "=" * 60)
    print("  Train / Val Bolumu")
    print("=" * 60)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    img_files = sorted(f for f in src_images.iterdir() if f.suffix.lower() in exts)
    if not img_files:
        print("  [!] Gorsel bulunamadi, bolme atlaniyor.")
        return

    random.shuffle(img_files)
    n_val = max(1, int(len(img_files) * val_ratio))
    val_files = img_files[:n_val]
    train_files = img_files[n_val:]

    for split_name, file_list in [("train", train_files), ("val", val_files)]:
        img_dst = dst_root / "images" / split_name
        lbl_dst = dst_root / "labels" / split_name
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_path in file_list:
            shutil.copy2(img_path, img_dst / img_path.name)
            # YOLO label (.txt)
            txt = src_labels / f"{img_path.stem}.txt"
            if txt.exists():
                shutil.copy2(txt, lbl_dst / txt.name)

        print(f"  {split_name:5s}: {len(file_list)} gorsel")

    print("  Bolme tamamlandi.\n")


def main():
    parser = argparse.ArgumentParser(description="QuakeVision pipeline")
    parser.add_argument("--skip-patch", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--sanity-check", action="store_true", default=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patch-size", type=int, default=512)
    args = parser.parse_args()

    raw_images = PROJECT_ROOT / "data" / "raw" / "images"
    raw_labels = PROJECT_ROOT / "data" / "raw" / "labels"
    patched = PROJECT_ROOT / "data" / "patched"
    processed = PROJECT_ROOT / "data" / "processed"
    sanity_dir = PROJECT_ROOT / "data" / "sanity_check"

    # 1. Dizin yapisi
    setup_directories()

    # 2. Patching
    if not args.skip_patch:
        print("\n" + "=" * 60)
        print("  Adim 2: Gorsel Parcalama")
        print("=" * 60)
        run_patching(raw_images, raw_labels, patched, args.patch_size)
    else:
        print("\n  [i] Patching adimi atlandi.")

    # 3. JSON -> YOLO
    if not args.skip_convert:
        print("\n" + "=" * 60)
        print("  Adim 3: JSON -> YOLOv8-Seg Donusumu")
        print("=" * 60)
        patched_labels = patched / "labels"
        yolo_labels_tmp = patched / "yolo_labels"
        run_conversion(patched_labels, yolo_labels_tmp, args.patch_size)

        # 3.5 Train/Val bolumu
        split_train_val(
            patched / "images", yolo_labels_tmp, processed, args.val_ratio
        )
    else:
        print("\n  [i] Donusum adimi atlandi.")

    # 4. Sanity Check
    if args.sanity_check:
        print("\n" + "=" * 60)
        print("  Adim 4: Sanity Check")
        print("=" * 60)
        sanity_check(
            processed / "images" / "train",
            processed / "labels" / "train",
            sanity_dir,
            args.patch_size,
            5,
        )

    # YAML dosyasi olustur
    yaml_path = PROJECT_ROOT / "data" / "quakevision.yaml"
    yaml_content = f"""# QuakeVision YOLOv8-Seg Dataset Config
path: {processed.as_posix()}
train: images/train
val: images/val

names:
  0: no-damage
  1: minor-damage
  2: major-damage
  3: destroyed
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\n  YAML config olusturuldu: {yaml_path}")
    print("\n" + "=" * 60)
    print("  Pipeline tamamlandi!")
    print("=" * 60)


if __name__ == "__main__":
    main()
