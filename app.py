import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import torch
from ultralytics import YOLO
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from fpdf import FPDF
from datetime import datetime

# 1. Arayüz Temel Yapısı
st.set_page_config(page_title="QuakeVision Dashboard", layout="wide", page_icon="🌍")

def create_pdf_report(df, total_b, destroyed_b, heavy_ratio):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=15)
    pdf.cell(200, 10, txt="QuakeVision Hasar Tespit Raporu", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 10, txt=f"Toplam Bina: {total_b}", ln=True)
    pdf.cell(200, 10, txt=f"Yikilan Bina: {destroyed_b}", ln=True)
    pdf.cell(200, 10, txt=f"Agir Hasar / Yikilmis Orani: %{heavy_ratio:.2f}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(30, 10, txt="Bina ID", border=1)
    pdf.cell(50, 10, txt="Hasar Sinifi", border=1)
    pdf.cell(40, 10, txt="Guven Skoru", border=1)
    pdf.cell(40, 10, txt="Alan (px)", border=1)
    pdf.ln()
    
    for _, row in df.iterrows():
        pdf.cell(30, 10, txt=str(row['Bina ID']), border=1)
        cls_name = str(row['Hasar Sınıfı']).replace("ı", "i").replace("ş", "s").replace("ğ", "g").replace("Ö", "O").replace("ç", "c")
        pdf.cell(50, 10, txt=cls_name, border=1)
        pdf.cell(40, 10, txt=f"{row['Güven Skoru']:.2f}", border=1)
        pdf.cell(40, 10, txt=str(int(row['Tahmini Alan (px)'])), border=1)
        pdf.ln()
        
    return pdf.output(dest="S").encode("latin-1")

# Model Yükleme (Cache)
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Poligon çizimi (Özel renklerle)
def draw_custom_masks(image, results):
    img_array = np.array(image).copy()
    
    # Renk Paleti (RGB)
    colors = {
        0: (0, 255, 0),    # 🟢 Yeşil: Sağlam
        1: (255, 255, 0),  # 🟡 Sarı: Hafif Hasarlı
        2: (255, 165, 0),  # 🟠 Turuncu: Ağır Hasarlı
        3: (255, 0, 0)     # 🔴 Kırmızı: Yıkılmış
    }
    
    for r in results:
        if r.masks is not None:
            masks = r.masks.xy
            classes = r.boxes.cls.cpu().numpy()
            
            for mask, cls in zip(masks, classes):
                color = colors.get(int(cls), (255, 255, 255))
                mask_points = np.array(mask, dtype=np.int32)
                
                cv2.polylines(img_array, [mask_points], isClosed=True, color=color, thickness=2)
                overlay = img_array.copy()
                cv2.fillPoly(overlay, [mask_points], color=color)
                cv2.addWeighted(overlay, 0.4, img_array, 0.6, 0, img_array)
                
    return Image.fromarray(img_array)

# YAN PANEL (SIDEBAR)
st.sidebar.title("⚙️ Model Ayarları")

default_model = os.path.join("models", "best.pt")
model_options = [default_model, "best_yolov8s.pt", "turkey_finetuned.pt"]

selected_model_path = st.sidebar.selectbox("Model Seçimi", model_options)
conf_threshold = st.sidebar.slider("Güven Eşiği (Confidence)", min_value=0.01, max_value=0.90, value=0.15, step=0.01)
iou_threshold = st.sidebar.slider("IoU Eşiği", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Renk Kodları
* 🟢 **Yeşil:** Sağlam
* 🟡 **Sarı:** Hafif Hasarlı
* 🟠 **Turuncu:** Ağır Hasarlı
* 🔴 **Kırmızı:** Yıkılmış
""")

try:
    if os.path.exists(selected_model_path):
        model = load_model(selected_model_path)
    else:
        st.sidebar.error(f"Seçilen model bulunamadı: {selected_model_path}")
        model = None
except Exception as e:
    st.sidebar.error(f"Model yüklenirken hata oluştu: {str(e)}")
    model = None

# ANA EKRAN
st.title("🌍 QuakeVision: Yapay Zeka Destekli Deprem Hasar Analiz Platformu")
st.markdown("AFAD ve karar vericiler için analitik grafikler, GIS destekli harita ve bina hasar raporlaması.")

st.subheader("Hızlı Test & Veri Yükleme")
use_sample = st.button("🇹🇷 Örnek Türkiye Verisi Deneme (Kahramanmaraş)")
uploaded_file = st.file_uploader("Veya kendi uydu görüntünüzü yükleyin (Deprem Öncesi / Sonrası)", type=["jpg", "png", "tif", "jpeg"])

image_to_process = None

if use_sample:
    sample_dir = os.path.join("data", "sample_turkey")
    if os.path.exists(sample_dir):
        samples = os.listdir(sample_dir)
        if samples:
            image_to_process = Image.open(os.path.join(sample_dir, samples[0])).convert("RGB")
            st.info("Türkiye (Kahramanmaraş) örnek verisi yüklendi.")
        else:
            st.warning(f"Örnek klasörü boş: {sample_dir}")
    else:
        st.warning(f"Örnek klasörü bulunamadı: {sample_dir}")
elif uploaded_file is not None:
    try:
        image_to_process = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Dosya yüklenirken hata oluştu: {e}")

if image_to_process is not None:
    if model is None:
        st.error("Model yüklenmediği için analiz yapılamıyor!")
    else:
        st.write("---")
        st.write("### 👁️ Görüntü Analizi")
        
        with st.spinner("Yapay zeka görüntüyü analiz ediyor..."):
            results = model.predict(
                source=image_to_process, 
                conf=conf_threshold, 
                iou=iou_threshold,
                imgsz=1536, 
                device=0 if torch.cuda.is_available() else "cpu"
            )
            result_image = draw_custom_masks(image_to_process, results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image_to_process, caption="Orijinal Uydu Görüntüsü", use_container_width=True)
        with col2:
            st.image(result_image, caption="QuakeVision Tahmini (AI Sonucu)", use_container_width=True)
            
        st.success("Analiz tamamlandı!")
        
        # Veri Çıkartma ve İstatistikler
        results_data = []
        bina_id = 1
        
        # Sınıf isimleri
        class_labels = {0: "Sağlam", 1: "Hafif Hasarlı", 2: "Ağır Hasarlı", 3: "Yıkılmış"}
        colors_hex = {0: "#00FF00", 1: "#FFFF00", 2: "#FFA500", 3: "#FF0000"}
        
        for r in results:
            if r.boxes is not None and r.masks is not None:
                classes = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                masks = r.masks.xy
                
                for mask, cls, conf in zip(masks, classes, confs):
                    cls_id = int(cls)
                    # Shoelace formula ile alan hesaplama
                    x = mask[:, 0]
                    y = mask[:, 1]
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    cx, cy = np.mean(x), np.mean(y)
                    
                    results_data.append({
                        "Bina ID": bina_id,
                        "Hasar Sınıfı": class_labels.get(cls_id, "Bilinmiyor"),
                        "Class_ID": cls_id,
                        "Güven Skoru": float(conf),
                        "Tahmini Alan (px)": area,
                        "cx": cx,
                        "cy": cy
                    })
                    bina_id += 1
                    
        df = pd.DataFrame(results_data)
        
        if not df.empty:
            st.write("---")
            st.write("### 📊 Analitik Karar Destek Paneli (Dashboard)")
            
            total_buildings = len(df)
            destroyed = len(df[df["Class_ID"] == 3])
            heavy = len(df[df["Class_ID"] == 2])
            heavy_ratio = ((destroyed + heavy) / total_buildings) * 100 if total_buildings > 0 else 0
            
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Toplam Bina Sayısı", total_buildings)
            col_m2.metric("Yıkılan Bina Sayısı", destroyed)
            col_m3.metric("Ağır Hasar / Yıkılmış Oranı", f"%{heavy_ratio:.1f}")
            
            st.write("#### Hasar Sınıfları Dağılımı")
            col_c1, col_c2 = st.columns(2)
            
            pie_data = df["Hasar Sınıfı"].value_counts().reset_index()
            pie_data.columns = ["Hasar Sınıfı", "Sayi"]
            
            fig_pie = px.pie(pie_data, values="Sayi", names="Hasar Sınıfı", 
                             color="Hasar Sınıfı", color_discrete_map={
                                 "Sağlam": "#00FF00", "Hafif Hasarlı": "#FFFF00", 
                                 "Ağır Hasarlı": "#FFA500", "Yıkılmış": "#FF0000"}, hole=0.3)
            col_c1.plotly_chart(fig_pie, use_container_width=True)
            
            fig_bar = px.bar(pie_data, x="Hasar Sınıfı", y="Sayi", 
                             color="Hasar Sınıfı", color_discrete_map={
                                 "Sağlam": "#00FF00", "Hafif Hasarlı": "#FFFF00", 
                                 "Ağır Hasarlı": "#FFA500", "Yıkılmış": "#FF0000"})
            col_c2.plotly_chart(fig_bar, use_container_width=True)
            
            st.write("---")
            st.write("### 🗺️ CBS (GIS) Harita Görünümü")
            
            map_center = [37.5753, 36.9228] # Kahramanmaraş varsayılan
            m = folium.Map(location=map_center, zoom_start=17, tiles="CartoDB dark_matter")
            
            img_w, img_h = image_to_process.size
            for _, row in df.iterrows():
                nx = row['cx'] / img_w
                ny = row['cy'] / img_h
                lat = map_center[0] - (ny - 0.5) * 0.003
                lon = map_center[1] + (nx - 0.5) * 0.003
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    color=colors_hex[row['Class_ID']],
                    fill=True,
                    fill_color=colors_hex[row['Class_ID']],
                    fill_opacity=0.8,
                    popup=f"Bina {row['Bina ID']} - {row['Hasar Sınıfı']} (Skor: {row['Güven Skoru']:.2f})"
                ).add_to(m)
                
            st_folium(m, width="100%", height=500)
            
            st.write("---")
            st.write("### 📄 Raporlama ve Dışa Aktarım")
            col_d1, col_d2 = st.columns(2)
            
            csv_data = df.drop(columns=["Class_ID", "cx", "cy"]).to_csv(index=False).encode('utf-8')
            col_d1.download_button("📥 Tabloyu Excel/CSV İndir", data=csv_data, file_name="hasar_raporu.csv", mime="text/csv")
            
            try:
                pdf_bytes = create_pdf_report(df, total_buildings, destroyed, heavy_ratio)
                col_d2.download_button("📄 Özet Hasar Raporu Oluştur (PDF)", data=pdf_bytes, file_name="hasar_ozet_raporu.pdf", mime="application/pdf")
            except Exception as e:
                col_d2.error("PDF oluşturulurken hata oluştu. 'fpdf' kütüphanesini kontrol edin.")
        else:
            st.warning("Bu görüntüde herhangi bir bina tespit edilemedi.")
