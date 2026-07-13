"""
QuakeVision - Dizin Yapısı Kurulumu
====================================
xBD deprem veri seti için gerekli klasör yapısını oluşturur.

Kullanım:
    python setup_dirs.py
"""

import os
import sys
from pathlib import Path


# Proje kök dizini (bu betiğin bulunduğu yer)
PROJECT_ROOT = Path(__file__).resolve().parent

# Oluşturulacak dizinler
DIRECTORIES = [
    # Ham veri (xBD orijinalleri)
    "data/raw/images",
    "data/raw/labels",
    # İşlenmiş veri — YOLOv8-Seg formatı
    "data/processed/images/train",
    "data/processed/images/val",
    "data/processed/labels/train",
    "data/processed/labels/val",
    # Doğrulama görselleri
    "data/sanity_check",
    # Betikler klasörü
    "scripts",
]


def setup_directories() -> None:
    """Proje dizin yapısını oluşturur."""
    print("=" * 60)
    print("  QuakeVision - Dizin Yapisi Kurulumu")
    print("=" * 60)
    print(f"\n  Proje kok dizini: {PROJECT_ROOT}\n")

    created_count = 0
    existing_count = 0

    for rel_path in DIRECTORIES:
        full_path = PROJECT_ROOT / rel_path
        try:
            if full_path.exists():
                print(f"  [OK] Zaten mevcut : {rel_path}")
                existing_count += 1
            else:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"  [+] Olusturuldu  : {rel_path}")
                created_count += 1
        except OSError as exc:
            print(f"  [X] HATA         : {rel_path} -> {exc}", file=sys.stderr)

    print(f"\n  Toplam: {created_count} yeni klasor olusturuldu, "
          f"{existing_count} klasor zaten mevcuttu.")
    print("=" * 60)


if __name__ == "__main__":
    setup_directories()
