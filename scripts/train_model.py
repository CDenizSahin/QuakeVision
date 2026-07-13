from ultralytics import YOLO
import torch
import os

def main():
    print("="*50)
    print("QuakeVision - Model Eğitimi (Görev 2.2)")
    print("="*50)
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("UYARI: GPU bulunamadı! Eğitim CPU üzerinde yapılacak ve çok yavaş olacaktır.")
    else:
        print(f"GPU Kullanılıyor: {torch.cuda.get_device_name(0)}")
    
    # 2. Haftadaki hedefe tam uygun parametreler:
    # Model: yolov8s-seg.pt (Small)
    print("Model yükleniyor: yolov8s-seg.pt (Small pre-trained)")
    model = YOLO("yolov8s-seg.pt")
    
    print("Eğitim başlatılıyor...")
    model.train(
        data="dataset.yaml", # 1. maddedeki yapılandırma
        epochs=30,           # Hedefteki 30 epoch
        patience=5,          # Early stopping: 5
        imgsz=512,           # Image Size 512
        batch=16,            # Batch size 16
        optimizer="AdamW",   # İstenen optimizer
        lr0=0.001,           # İstenen learning rate
        mosaic=0.5,          # Data Augmentation
        fliplr=0.5,          # Data Augmentation
        device=0 if torch.cuda.is_available() else "cpu",
        project="runs/segment",
        name="quakevision_s" # Karışmaması için _s (small) ekledik
    )
    print("Eğitim tamamlandı!")

if __name__ == "__main__":
    main()
