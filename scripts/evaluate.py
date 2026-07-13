from ultralytics import YOLO
import torch
import os
import shutil

def main():
    print("="*50)
    print("QuakeVision - Değerlendirme ve Raporlama (Görev 2.3)")
    print("="*50)
    
    # Otomatik CPU/GPU fallback
    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("UYARI: GPU bulunamadı, değerlendirme CPU ile yapılacak!")
    
    # Bizim daha önceden (veya şimdi) eğittiğimiz modelin ağırlıkları
    # Biz önceki aşamada "quakevision" ismiyle eğittiğimiz için onun weights klasörünü kullanıyoruz.
    best_model_path = os.path.join("runs", "segment", "quakevision", "weights", "best.pt")
    
    if not os.path.exists(best_model_path):
        print(f"HATA: Model bulunamadı ({best_model_path})")
        print("Lütfen önce modeli eğitin (train_model.py)")
        return
        
    print(f"Model yükleniyor: {best_model_path}")
    model = YOLO(best_model_path)
    
    print("\nDoğrulama (Validation) başlatılıyor, lütfen bekleyin...")
    # mAP değerlerini ve grafikleri hesaplar
    metrics = model.val(
        data="dataset.yaml",
        device=device,
        plots=True # Matrix ve Eğrilerin çizilmesi için
    )
    
    print("\n" + "="*20 + " SONUÇLAR " + "="*20)
    print(f"mAP@0.5      : {metrics.seg.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.seg.map:.4f}")
    print("="*50)
    
    # 3. İstenilen rapor klasörüne dosyaları çıkarma
    reports_dir = os.path.join("reports", "figures")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Ultralytics sonuçları metrics.save_dir klasörüne (örn. runs/segment/val) çıkarır.
    # Görevde özellikle istenilen Confusion Matrix ve F1 curve grafiklerini alıp kopyalıyoruz.
    files_to_copy = ["confusion_matrix_normalized.png", "MaskF1_curve.png", "MaskPR_curve.png"]
    
    for file_name in files_to_copy:
        # Önce save_dir (yeni val klasörü) içine bakalım
        src = os.path.join(metrics.save_dir, file_name)
        dst = os.path.join(reports_dir, file_name)
        
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Rapor Kaydedildi: {dst}")
        else:
            # Eğer oradaysa eski quakevision eğitim klasöründe olabilir
            alt_src = os.path.join("runs", "segment", "quakevision", file_name)
            if os.path.exists(alt_src):
                shutil.copy(alt_src, dst)
                print(f"Rapor Kaydedildi: {dst}")
            else:
                print(f"Uyarı: {file_name} bulunamadı.")
                
    print("\nTüm değerlendirme süreçleri başarıyla tamamlandı!")

if __name__ == "__main__":
    main()
