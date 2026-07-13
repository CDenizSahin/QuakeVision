"""
QuakeVision - xView2 Veri Seti Hazirlama
==========================================
xView2.org'dan indirilen .tar / .tgz arsivlerini acar ve
pipeline'in bekledigini dizin yapisina kopyalar.

Indirilen dosyalar:
    train_images_labels_targets.tar   (~7.8 GB)
    test_images_labels_targets.tar    (~2.6 GB)
    xview_geotransforms.json.tgz      (~500 KB)

Hedef yapi:
    data/raw/images/   (sadece post_disaster .png)
    data/raw/labels/   (sadece post_disaster .json)

Kullanim:
    python scripts/prepare_kaggle_data.py
"""

from __future__ import annotations
import argparse, shutil, sys, tarfile, os, glob
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Varsayilan indirme konumu
DEFAULT_DOWNLOADS = Path.home() / "Downloads"


def extract_tar(tar_path: Path, dest: Path, desc: str = "Cikariliyor") -> None:
    """Tar arsivini ilerleme cubugu ile cikarir."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n  Arsiv  : {tar_path.name}")
    print(f"  Hedef  : {dest}")

    try:
        # Magic bytes ile gzip mi kontrol et (uzanti yaniltici olabilir)
        with open(str(tar_path), "rb") as fcheck:
            is_gzip = fcheck.read(2) == b"\x1f\x8b"
        mode = "r:gz" if is_gzip else "r:"
        with tarfile.open(str(tar_path), mode) as tar:
            members = tar.getmembers()
            print(f"  Dosya  : {len(members)} adet")
            for member in tqdm(members, desc=f"  {desc}", unit="file"):
                try:
                    tar.extract(member, path=str(dest), filter="data")
                except (tarfile.TarError, OSError, KeyError) as e:
                    # Bazi dosyalar sorunlu olabilir, atla
                    pass
    except tarfile.TarError as e:
        print(f"  [!] Arsiv hatasi: {e}", file=sys.stderr)
        sys.exit(1)


def find_and_copy_post_disaster(
    extracted_dir: Path,
    output_images: Path,
    output_labels: Path,
) -> tuple[int, int]:
    """
    Cikarilmis dizinden post_disaster dosyalarini bulup kopyalar.

    xView2 arsiv yapisi (cikarildiktan sonra):
        train/
            images/
                <disaster>_<id>_pre_disaster.png
                <disaster>_<id>_post_disaster.png
            labels/
                <disaster>_<id>_pre_disaster.json
                <disaster>_<id>_post_disaster.json
            targets/
                ...
    """
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    # Post-disaster gorselleri bul (recursive arama)
    img_pattern = "**/*_post_disaster.png"
    lbl_pattern = "**/*_post_disaster.json"

    img_files = sorted(extracted_dir.glob(img_pattern))
    lbl_files = sorted(extracted_dir.glob(lbl_pattern))

    print(f"\n  Bulunan post_disaster gorseller : {len(img_files)}")
    print(f"  Bulunan post_disaster etiketler : {len(lbl_files)}")

    copied_img = 0
    for f in tqdm(img_files, desc="  Gorseller", unit="file"):
        dst = output_images / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            copied_img += 1

    copied_lbl = 0
    for f in tqdm(lbl_files, desc="  Etiketler", unit="file"):
        dst = output_labels / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
            copied_lbl += 1

    return copied_img, copied_lbl


def main():
    parser = argparse.ArgumentParser(
        description="xView2 arsivlerini ac ve pipeline'a hazirla"
    )
    parser.add_argument(
        "--downloads-dir", type=Path, default=DEFAULT_DOWNLOADS,
        help=f"Indirilen dosyalarin bulundugu klasor (varsayilan: {DEFAULT_DOWNLOADS})"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Arsiv acma adimini atla (zaten acilmissa)"
    )
    args = parser.parse_args()

    dl = args.downloads_dir
    raw_dir = PROJECT_ROOT / "data" / "raw"
    extract_base = PROJECT_ROOT / "data" / "extracted"

    print("=" * 60)
    print("  QuakeVision - xView2 Veri Seti Hazirlama")
    print("=" * 60)

    # --- 1. Arsivleri cikar ---
    train_tar = dl / "train_images_labels_targets.tar"
    test_tar = dl / "test_images_labels_targets.tar"
    geo_tgz = dl / "xview_geotransforms.json.tgz"

    if not args.skip_extract:
        # Train arsivi
        if train_tar.exists():
            extract_tar(train_tar, extract_base / "train", "Train cikariliyor")
        else:
            print(f"\n  [!] Train arsivi bulunamadi: {train_tar}")

        # Test arsivi
        if test_tar.exists():
            extract_tar(test_tar, extract_base / "test", "Test cikariliyor")
        else:
            print(f"\n  [!] Test arsivi bulunamadi: {test_tar}")

        # Geotransforms
        if geo_tgz.exists():
            extract_tar(geo_tgz, extract_base / "geo", "Geotransforms")
        else:
            print(f"\n  [i] Geotransforms bulunamadi (opsiyonel)")
    else:
        print("\n  [i] Arsiv acma adimi atlandi (--skip-extract)")

    # --- 2. Post-disaster dosyalarini data/raw/ altina kopyala ---
    print("\n" + "=" * 60)
    print("  Post-disaster dosyalari kopyalaniyor...")
    print("=" * 60)

    train_extracted = extract_base / "train"
    if train_extracted.exists():
        img_c, lbl_c = find_and_copy_post_disaster(
            train_extracted,
            raw_dir / "images",
            raw_dir / "labels",
        )
        print(f"\n  Kopyalanan gorsel : {img_c}")
        print(f"  Kopyalanan etiket: {lbl_c}")
    else:
        print("  [!] Train cikarilmis dizini bulunamadi!")

    # --- 3. Ozet ---
    img_count = len(list((raw_dir / "images").glob("*.png"))) if (raw_dir / "images").exists() else 0
    lbl_count = len(list((raw_dir / "labels").glob("*.json"))) if (raw_dir / "labels").exists() else 0

    print("\n" + "=" * 60)
    print("  Hazirlik tamamlandi!")
    print(f"  data/raw/images/ : {img_count} gorsel")
    print(f"  data/raw/labels/ : {lbl_count} etiket")
    print("  Simdi 'py run_pipeline.py' ile devam edin.")
    print("=" * 60)


if __name__ == "__main__":
    main()
