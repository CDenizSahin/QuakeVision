import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import io
import requests
import pandas as pd
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from fpdf import FPDF
from datetime import datetime

API_URL = "http://127.0.0.1:8020/analyze/image"

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

def draw_custom_masks_from_json(image, detections):
    img_array = np.array(image).copy()
    
    colors = {
        0: (0, 255, 0),    # 🟢 Yeşil: Sağlam
        1: (255, 255, 0),  # 🟡 Sarı: Hafif Hasarlı
        2: (255, 165, 0),  # 🟠 Turuncu: Ağır Hasarlı
        3: (255, 0, 0)     # 🔴 Kırmızı: Yıkılmış
    }
    
    for det in detections:
        cls_id = det["class_id"]
        color = colors.get(cls_id, (255, 255, 255))
        polygon = np.array(det["polygon"], dtype=np.int32)
        
        cv2.polylines(img_array, [polygon], isClosed=True, color=color, thickness=2)
        overlay = img_array.copy()
        cv2.fillPoly(overlay, [polygon], color=color)
        cv2.addWeighted(overlay, 0.4, img_array, 0.6, 0, img_array)
                
    return Image.fromarray(img_array)

# YAN PANEL (SIDEBAR)
st.sidebar.title("⚙️ Model & API Ayarları")

st.sidebar.success("✅ FastAPI Backend Bağlantısı Aktif")
conf_threshold = st.sidebar.slider("Güven Eşiği (Confidence)", min_value=0.01, max_value=0.90, value=0.15, step=0.01)
iou_threshold = st.sidebar.slider("IoU Eşiği", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Harita (GIS) Ayarları")
st.sidebar.info("Yüklediğiniz görüntünün gerçek konumunu haritada görmek için koordinatları buraya girebilirsiniz.")
map_lat = st.sidebar.number_input("Enlem (Latitude)", value=37.5753, format="%.6f")
map_lon = st.sidebar.number_input("Boylam (Longitude)", value=36.9228, format="%.6f")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Renk Kodları
* 🟢 **Yeşil:** Sağlam
* 🟡 **Sarı:** Hafif Hasarlı
* 🟠 **Turuncu:** Ağır Hasarlı
* 🔴 **Kırmızı:** Yıkılmış
""")

# ANA EKRAN
st.title("🌍 QuakeVision Enterprise: AI & GIS Platform")
st.markdown("AFAD ve karar vericiler için analitik grafikler, GIS destekli Yoğunluk Haritası (Heatmap) ve bina hasar raporlaması.")

st.subheader("Hızlı Test & Veri Yükleme")
use_sample = st.button("🇹🇷 Örnek Türkiye Verisi Deneme (Kahramanmaraş)")
uploaded_file = st.file_uploader("Veya kendi uydu görüntünüzü yükleyin", type=["jpg", "png", "tif", "jpeg"])

image_to_process = None
image_bytes = None

if use_sample:
    sample_dir = os.path.join("data", "sample_turkey")
    if os.path.exists(sample_dir):
        samples = os.listdir(sample_dir)
        if samples:
            img_path = os.path.join(sample_dir, samples[0])
            image_to_process = Image.open(img_path).convert("RGB")
            with open(img_path, "rb") as f:
                image_bytes = f.read()
            st.info("Türkiye (Kahramanmaraş) örnek verisi yüklendi.")
        else:
            st.warning(f"Örnek klasörü boş: {sample_dir}")
    else:
        st.warning(f"Örnek klasörü bulunamadı: {sample_dir}")
elif uploaded_file is not None:
    try:
        image_to_process = Image.open(uploaded_file).convert("RGB")
        image_bytes = uploaded_file.getvalue()
    except Exception as e:
        st.error(f"Dosya yüklenirken hata oluştu: {e}")

if image_to_process is not None and image_bytes is not None:
    st.write("---")
    st.write("### 👁️ Görüntü Analizi")
    
    with st.spinner("Görüntü FastAPI Sunucusunda Analiz Ediliyor..."):
        try:
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
            data = {"conf_threshold": conf_threshold, "iou_threshold": iou_threshold}
            
            response = requests.post(API_URL, files=files, data=data)
            
            if response.status_code == 200:
                results_json = response.json()
                detections = results_json.get("detections", [])
                
                result_image = draw_custom_masks_from_json(image_to_process, detections)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_to_process, caption="Orijinal Uydu Görüntüsü", use_container_width=True)
                with col2:
                    st.image(result_image, caption="QuakeVision Tahmini (AI Sonucu)", use_container_width=True)
                    
                st.success(f"Analiz tamamlandı! {len(detections)} adet bina tespit edildi.")
                
                # Veri Çıkartma ve İstatistikler
                results_data = []
                bina_id = 1
                for det in detections:
                    results_data.append({
                        "Bina ID": bina_id,
                        "Hasar Sınıfı": det["class_name"],
                        "Class_ID": det["class_id"],
                        "Güven Skoru": det["confidence"],
                        "Tahmini Alan (px)": det["area_px"],
                        "cx": det["cx"],
                        "cy": det["cy"]
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
                    
                    colors_hex = {0: "#00FF00", 1: "#FFFF00", 2: "#FFA500", 3: "#FF0000"}
                    
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
                    st.write("### 🗺️ CBS (GIS) Isı Haritası (Heatmap) Görünümü")
                    
                    map_center = [map_lat, map_lon]
                    
                    # EXIF Okuma
                    try:
                        exif_info = image_to_process.getexif()
                        if exif_info:
                            from PIL.ExifTags import TAGS, GPSTAGS
                            for tag, value in exif_info.items():
                                if TAGS.get(tag, tag) == "GPSInfo":
                                    gps_lat = value.get(2)
                                    gps_lat_ref = value.get(1)
                                    gps_lon = value.get(4)
                                    gps_lon_ref = value.get(3)
                                    
                                    if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
                                        def to_deg(v):
                                            return float(v[0]) + (float(v[1]) / 60.0) + (float(v[2]) / 3600.0)
                                        
                                        lat = to_deg(gps_lat)
                                        if gps_lat_ref != "N": lat = -lat
                                        lon = to_deg(gps_lon)
                                        if gps_lon_ref != "E": lon = -lon
                                        
                                        map_center = [lat, lon]
                                        st.success(f"📸 Görüntü içindeki GPS verisi otomatik bulundu: {lat:.6f}, {lon:.6f}.")
                                        break
                    except Exception:
                        pass
                        
                    m = folium.Map(location=map_center, zoom_start=17, tiles="CartoDB dark_matter")
                    
                    img_w = results_json["width"]
                    img_h = results_json["height"]
                    
                    heat_data = []
                    
                    for _, row in df.iterrows():
                        nx = row['cx'] / img_w
                        ny = row['cy'] / img_h
                        lat = map_center[0] - (ny - 0.5) * 0.003
                        lon = map_center[1] + (nx - 0.5) * 0.003
                        
                        # Hasara göre HeatMap ağırlığı (weight)
                        weight = 0.2
                        if row['Class_ID'] == 3: weight = 1.0
                        elif row['Class_ID'] == 2: weight = 0.7
                        elif row['Class_ID'] == 1: weight = 0.4
                        
                        heat_data.append([lat, lon, weight])
                        
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=3,
                            color=colors_hex[row['Class_ID']],
                            fill=True,
                            fill_color=colors_hex[row['Class_ID']],
                            fill_opacity=0.8,
                            popup=f"Bina {row['Bina ID']} - {row['Hasar Sınıfı']}"
                        ).add_to(m)
                        
                    # HeatMap Ekleme
                    HeatMap(heat_data, min_opacity=0.4, radius=20, blur=15, max_zoom=1).add_to(m)
                        
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
                        col_d2.error("PDF oluşturulurken hata oluştu.")
                        
                    st.write("---")
                    st.write("### 🤖 Yapay Zeka (LLM) Afet Durum Raporu")
                    if st.button("Llama 3 ile Afet Durum Raporu Üret"):
                        with st.spinner("Ollama (Llama 3) raporu yazıyor, lütfen bekleyin..."):
                            report_req_data = {
                                "total_buildings": total_buildings,
                                "destroyed": destroyed,
                                "heavy_damage": heavy,
                                "light_damage": len(df[df["Class_ID"] == 1]),
                                "intact": len(df[df["Class_ID"] == 0]),
                                "model_name": "llama3"
                            }
                            try:
                                report_resp = requests.post(f"http://127.0.0.1:8020/generate_report", json=report_req_data)
                                if report_resp.status_code == 200:
                                    report_text = report_resp.json().get("report", "")
                                    st.success("Rapor başarıyla üretildi!")
                                    st.markdown(report_text)
                                else:
                                    st.error(f"Rapor üretilemedi. Hata: {report_resp.text}")
                            except Exception as e:
                                st.error(f"API'ye ulaşılamadı: {e}")
                else:
                    st.warning("Bu görüntüde herhangi bir bina tespit edilemedi.")
            else:
                try:
                    error_msg = response.json().get("message", response.text)
                except:
                    error_msg = response.text
                st.error(f"API Hatası: {error_msg}")
        except requests.exceptions.ConnectionError:
            st.error("FastAPI Sunucusuna ulaşılamıyor! Lütfen 'uvicorn api.main:app' komutunun çalıştığından emin olun.")
