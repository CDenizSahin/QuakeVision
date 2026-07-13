"""
QuakeVision - YOLOv8 Instance Segmentation Egitimi
====================================================
Hazirlanmis veri seti ile YOLOv8-Seg modelini egitir.

GPU: NVIDIA RTX 4050 Laptop (6 GB VRAM)
Veri: 11,196 patch (512x512), 4 sinif

Kullanim:
    python train.py
    python train.py --model yolov8s-seg.pt --epochs 50
    python train.py --resume  (son checkpoint'tan devam)
"""

from __future__ import annotations
import argparse, sys, os
from pathlib import Path

# Proje koku
PROJECT_ROOT = Path(__file__).resolve().parent


def check_environment():
    """GPU ve kutuphane kontrolu."""
    import torch
    print("=" * 60)
    print("  Ortam Kontrolu")
    print("=" * 60)
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU      : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM     : {vram:.1f} GB")
    else:
        print("  GPU      : YOK (CPU ile egitim yapilacak)")
    print()


def train(args):
    """YOLOv8-Seg egitimini baslatir."""
    from ultralytics import YOLO

    check_environment()

    yaml_path = PROJECT_ROOT / "data" / "quakevision.yaml"
    if not yaml_path.exists():
        print(f"  [!] Dataset config bulunamadi: {yaml_path}")
        print("  Once 'py run_pipeline.py' calistirin.")
        sys.exit(1)

    print("=" * 60)
    print("  YOLOv8-Seg Egitimi Baslatiliyor")
    print("=" * 60)
    print(f"  Model   : {args.model}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  ImgSize : {args.imgsz}")
    print(f"  Batch   : {args.batch}")
    print(f"  Dataset : {yaml_path}")
    print()

    # Model yukle
    if args.resume:
        # Son checkpoint'tan devam
        last_pt = PROJECT_ROOT / "runs" / "segment" / "quakevision" / "weights" / "last.pt"
        if last_pt.exists():
            model = YOLO(str(last_pt))
            print(f"  Checkpoint'tan devam: {last_pt}")
        else:
            print(f"  [!] Checkpoint bulunamadi: {last_pt}")
            print("  Sifirdan baslaniyor...")
            model = YOLO(args.model)
    else:
        model = YOLO(args.model)

    # Egitim parametreleri (RTX 4050 6GB VRAM icin optimize)
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0 if _has_cuda() else "cpu",
        workers=args.workers,
        project=str(PROJECT_ROOT / "runs" / "segment"),
        name="quakevision",
        exist_ok=True,
        # Optimizasyon
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Augmentation (mosaic kapali - VRAM tasarrufu)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.0,
        # Kayit
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("  Egitim tamamlandi!")
    print(f"  Sonuclar: {PROJECT_ROOT / 'runs' / 'segment' / 'quakevision'}")
    print("=" * 60)

    # Dogrulama
    print("\n  Validation calistiriliyor...")
    val_results = model.val()
    print(f"  mAP50    : {val_results.seg.map50:.4f}")
    print(f"  mAP50-95 : {val_results.seg.map:.4f}")

    return results


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description="QuakeVision YOLOv8-Seg egitimi")
    parser.add_argument(
        "--model", type=str, default="yolov8n-seg.pt",
        help="Baslangic modeli (varsayilan: yolov8n-seg.pt = nano)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Egitim epoch sayisi (varsayilan: 100)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=512,
        help="Gorsel boyutu (varsayilan: 512)"
    )
    parser.add_argument(
        "--batch", type=int, default=4,
        help="Batch boyutu - RTX 4050 6GB icin 4 guvenli (varsayilan: 4)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader worker sayisi (varsayilan: 4)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Son checkpoint'tan devam et"
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
