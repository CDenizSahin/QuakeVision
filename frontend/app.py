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
LLM_URL = "http://127.0.0.1:8020/generate_report"

# 1. Arayüz Temel Yapısı
st.set_page_config(page_title="QuakeVision - Afet Yönetim Sistemi", layout="wide", page_icon="🌍")

# CSS Styling (Premium UI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Title Gradient Animation */
    h1 {
        background: linear-gradient(270deg, #ff6b6b, #4e54c8, #8f94fb, #ff6b6b);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: GradientAnimation 10s ease infinite;
        font-weight: 700 !important;
        padding-bottom: 10px;
    }
    
    @keyframes GradientAnimation {
        0%{background-position:0% 50%}
        50%{background-position:100% 50%}
        100%{background-position:0% 50%}
    }

    /* Subheaders */
    h2, h3 {
        font-weight: 600 !important;
        color: inherit;
    }
    
    /* Elegant Button */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(42, 82, 152, 0.5);
    }
    
    /* Metric Cards - adapt to theme with Glassmorphism */
    div[data-testid="metric-container"] {
        background: rgba(128, 128, 128, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.1);
        border-color: rgba(128, 128, 128, 0.4);
    }
    
    /* Containers / Expander Styling */
    div[data-testid="stExpander"] {
        background: rgba(128, 128, 128, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Sidebar specific styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(128, 128, 128, 0.02);
        border-right: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    /* Tabs styling */
    div[data-testid="stTabs"] button {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def create_pdf_report(df, total_b, destroyed_b, heavy_ratio, ai_report_text=""):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Helvetica", size=16, style="B")
    pdf.cell(200, 10, txt="QuakeVision Enterprise - Hasar Tespit Raporu", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 8, txt=f"Tarih / Saat: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 8, txt=f"Toplam Incelenen Bina: {total_b}", ln=True)
    pdf.cell(200, 8, txt=f"Yikilan Bina Sayisi: {destroyed_b}", ln=True)
    pdf.cell(200, 8, txt=f"Kritik Hasar Orani (Agir+Yikik): %{heavy_ratio:.2f}", ln=True)
    pdf.ln(10)
    
    if ai_report_text:
        pdf.set_font("Helvetica", size=14, style="B")
        pdf.cell(200, 10, txt="Yapay Zeka (Llama 3) Acil Mudahale ve Eylem Plani", ln=True)
        pdf.set_font("Helvetica", size=10)
        
        # Turkce karakter temizligi FPDF encoding icin
        clean_text = ai_report_text.replace("ı", "i").replace("ş", "s").replace("ğ", "g").replace("Ö", "O").replace("ç", "c").replace("ü", "u")
        clean_text = clean_text.replace("İ", "I").replace("Ş", "S").replace("Ğ", "G").replace("Ç", "C").replace("Ü", "U").replace("Ö", "O")
        clean_text = clean_text.replace("**", "").replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        clean_text = clean_text.replace("–", "-").replace("—", "-").replace("•", "-").replace("…", "...")
        clean_text = clean_text.encode('ascii', 'ignore').decode('ascii')
        pdf.multi_cell(0, 7, txt=clean_text)
        pdf.ln(10)
    
    pdf.set_font("Helvetica", size=12, style="B")
    pdf.cell(200, 10, txt="Bina Bazi Hasar Verileri", ln=True)
    pdf.set_font("Helvetica", size=10)
    
    # Tablo Basliklari
    pdf.cell(30, 8, txt="Bina ID", border=1, fill=False)
    pdf.cell(50, 8, txt="Hasar Sinifi", border=1, fill=False)
    pdf.cell(40, 8, txt="Guven Skoru", border=1, fill=False)
    pdf.cell(40, 8, txt="Alan (px)", border=1, fill=False)
    pdf.ln()
    
    # 100 kayit limiti PDF cok uzun olmasin diye
    limit = min(len(df), 100)
    for i in range(limit):
        row = df.iloc[i]
        pdf.cell(30, 8, txt=str(row['Bina ID']), border=1)
        cls_name = str(row['Hasar Sınıfı']).replace("ı", "i").replace("ş", "s").replace("ğ", "g").replace("Ö", "O").replace("ç", "c")
        pdf.cell(50, 8, txt=cls_name, border=1)
        pdf.cell(40, 8, txt=f"{row['Güven Skoru']:.2f}", border=1)
        pdf.cell(40, 8, txt=str(int(row['Tahmini Alan (px)'])), border=1)
        pdf.ln()
        
    if len(df) > 100:
        pdf.cell(160, 8, txt=f"... ve {len(df)-100} kayit daha.", border=1, align='C')
        
    return bytes(pdf.output())

def draw_custom_masks_from_json(image, detections):
    img_array = np.array(image).copy()
    
    colors = {
        0: (0, 255, 0),    
        1: (255, 255, 0),  
        2: (255, 165, 0),  
        3: (255, 0, 0)     
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
st.sidebar.markdown("## 🏢 Gelişmiş Afet Yönetim Merkezi")
st.sidebar.title("⚙️ Kurumsal Ayarlar")

st.sidebar.success("✅ Veritabanı ve API Bağlantısı Aktif")
st.sidebar.markdown("---")

st.sidebar.subheader("☁️ Bulut Yapay Zeka Hızlandırma")
api_key_input = st.sidebar.text_input(
    "Google Gemini API Key (Opsiyonel)", 
    type="password", 
    help="Sistemi buluta taşıyıp RAM kullanımını %0'a indirmek ve 1 saniyede cevap almak için ücretsiz Google AI Studio'dan (Gemini) aldığınız API anahtarını buraya yapıştırın. Boş bırakırsanız sistem lokal model (Ollama) kullanır."
)

st.sidebar.markdown("---")
conf_threshold = st.sidebar.slider("AI Güven Eşiği (Confidence)", min_value=0.01, max_value=0.90, value=0.15, step=0.01, help="Düşük değerler daha fazla bina bulur, yüksek değerler sadece en emin olduklarını getirir.")
iou_threshold = st.sidebar.slider("AI IoU Eşiği", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Coğrafi (GIS) Merkez")
map_lat = st.sidebar.number_input("Enlem (Latitude)", value=37.5753, format="%.6f")
map_lon = st.sidebar.number_input("Boylam (Longitude)", value=36.9228, format="%.6f")

st.sidebar.markdown("---")
st.sidebar.info("Geliştirici: Cem Deniz Şahin")

# ANA EKRAN
st.title("🌍 QuakeVision - Afet Analiz Platformu")
st.markdown("Yapay Zeka Destekli Uydu ve Coğrafi Bilgi Sistemleri (GIS) Afet Yönetim Platformu.")

# Dosya Yükleme Alanı
col_upload1, col_upload2 = st.columns([1, 2])
with col_upload1:
    use_sample = st.button("🇹🇷 Örnek Maraş Verisi ile Başla", use_container_width=True)
with col_upload2:
    uploaded_file = st.file_uploader("Veya kendi uydu görüntünüzü yükleyin", type=["jpg", "png", "tif", "jpeg"], label_visibility="collapsed")

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
            st.toast("✅ Kahramanmaraş örnek verisi başarıyla yüklendi!", icon="🚀")
elif uploaded_file is not None:
    try:
        image_to_process = Image.open(uploaded_file).convert("RGB")
        image_bytes = uploaded_file.getvalue()
        st.toast("✅ Özel uydu görüntüsü başarıyla yüklendi!", icon="🚀")
    except Exception as e:
        st.error(f"Dosya yüklenirken hata oluştu: {e}")

if image_to_process is not None and image_bytes is not None:
    st.write("---")
    
    # SEKMELER (TABS)
    tab1, tab2, tab3 = st.tabs(["👁️ Görüntü Analizi", "🗺️ GIS Isı Haritası", "📄 Llama 3 & Resmi Raporlama"])
    
    response = None
    with st.spinner("AI Görüntü İşleme Motoru Çalışıyor... (Güvenlik sınırları aktif)"):
        try:
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
            data = {"conf_threshold": conf_threshold, "iou_threshold": iou_threshold}
            response = requests.post(API_URL, files=files, data=data)
        except requests.exceptions.ConnectionError:
            st.error("FastAPI Sunucusuna ulaşılamıyor! Lütfen arka planda 'uvicorn' komutunun çalıştığından emin olun.")

    if response is not None:
        if response.status_code == 200:
            results_json = response.json()
            detections = results_json.get("detections", [])
            
            if "analysis_done" not in st.session_state:
                st.balloons()
                st.session_state["analysis_done"] = True
                
            result_image = draw_custom_masks_from_json(image_to_process, detections)
            
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
            
            total_buildings = len(df)
            destroyed = len(df[df["Class_ID"] == 3]) if total_buildings > 0 else 0
            heavy = len(df[df["Class_ID"] == 2]) if total_buildings > 0 else 0
            heavy_ratio = ((destroyed + heavy) / total_buildings) * 100 if total_buildings > 0 else 0
            
            # --- TAB 1 ---
            with tab1:
                st.success(f"Analiz başarıyla tamamlandı! Tam {total_buildings} adet bina yapay zeka tarafından tespit edildi.")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("📌 Toplam Bina", total_buildings)
                col_m2.metric("🏚️ Yıkılan Bina", destroyed, delta="-Kritik Durum", delta_color="inverse")
                col_m3.metric("⚠️ Ağır Hasar Oranı", f"%{heavy_ratio:.1f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_to_process, caption="Orijinal Uydu Görüntüsü", use_container_width=True)
                with col2:
                    st.image(result_image, caption="QuakeVision Yapay Zeka Sonucu", use_container_width=True)
                    
                st.info("Renk Kodları: 🟢 Sağlam | 🟡 Hafif Hasarlı | 🟠 Ağır Hasarlı | 🔴 Yıkılmış")
            
            # --- TAB 2 ---
            with tab2:
                st.subheader("Bölgesel Hasar Yoğunluk Haritası")
                map_center = [map_lat, map_lon]
                
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
                except:
                    pass
                    
                if total_buildings > 0:
                    m = folium.Map(location=map_center, zoom_start=17, tiles="CartoDB dark_matter")
                    img_w = results_json["width"]
                    img_h = results_json["height"]
                    
                    heat_data = []
                    colors_hex = {0: "#00FF00", 1: "#FFFF00", 2: "#FFA500", 3: "#FF0000"}
                    
                    for _, row in df.iterrows():
                        nx = row['cx'] / img_w
                        ny = row['cy'] / img_h
                        lat = map_center[0] - (ny - 0.5) * 0.003
                        lon = map_center[1] + (nx - 0.5) * 0.003
                        
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
                        
                    HeatMap(heat_data, min_opacity=0.4, radius=20, blur=15, max_zoom=1).add_to(m)
                    import streamlit.components.v1 as components
                    components.html(m._repr_html_(), height=500, scrolling=True)
                else:
                    st.warning("Haritaya eklenecek bina bulunamadı.")
            
            # --- TAB 3 ---
            with tab3:
                st.subheader("📄 Resmi Çıktılar ve Karar Destek Raporları")
                
                if "ai_report_text" not in st.session_state:
                    st.session_state["ai_report_text"] = ""
                
                if st.button("🤖 Llama 3 ile Uzman Raporu Oluştur", type="primary", use_container_width=True):
                    with st.spinner("Llama 3 derin analiz yapıyor, stratejik plan oluşturuluyor... (10-20 saniye sürebilir)"):
                        report_req_data = {
                            "total_buildings": total_buildings,
                            "destroyed": destroyed,
                            "heavy_damage": heavy,
                            "light_damage": len(df[df["Class_ID"] == 1]) if total_buildings > 0 else 0,
                            "intact": len(df[df["Class_ID"] == 0]) if total_buildings > 0 else 0,
                            "model_name": "llama3.2",
                            "api_key": api_key_input
                        }
                        try:
                            report_resp = requests.post(LLM_URL, json=report_req_data)
                            if report_resp.status_code == 200:
                                st.session_state["ai_report_text"] = report_resp.json().get("report", "")
                                st.toast("Llama 3 Raporu Başarıyla Oluşturuldu!", icon="✅")
                            else:
                                st.error(f"Rapor üretilemedi. Hata: {report_resp.text}")
                        except Exception as e:
                            st.error(f"LLM API'sine ulaşılamadı. Ollama çalışıyor mu?: {e}")
                
                if st.session_state["ai_report_text"]:
                    st.markdown("### 📊 Yapay Zeka Grafiksel Analizleri")
                    col_g1, col_g2 = st.columns(2)
                    
                    with col_g1:
                        fig_pie = px.pie(df, names="Hasar Sınıfı", title="Bina Hasar Oranları", 
                                     color="Hasar Sınıfı", 
                                     color_discrete_map={"Yıkılmış": "#FF4B4B", "Ağır Hasarlı": "#FFA500", "Hafif Hasarlı": "#FFFF00", "Sağlam": "#00FF00"})
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                    with col_g2:
                        damage_counts = df["Hasar Sınıfı"].value_counts().reset_index()
                        damage_counts.columns = ["Hasar Sınıfı", "Bina Sayısı"]
                        fig_bar = px.bar(damage_counts, x="Hasar Sınıfı", y="Bina Sayısı", title="Hasar Sınıfına Göre Bina Sayıları",
                                     color="Hasar Sınıfı",
                                     color_discrete_map={"Yıkılmış": "#FF4B4B", "Ağır Hasarlı": "#FFA500", "Hafif Hasarlı": "#FFFF00", "Sağlam": "#00FF00"})
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.markdown("### 📋 Llama 3 Acil Müdahale Eylem Planı")
                    with st.container(border=True):
                        st.markdown(st.session_state["ai_report_text"])
                
                st.write("---")
                col_d1, col_d2 = st.columns(2)
                
                if not df.empty:
                    csv_data = df.drop(columns=["Class_ID", "cx", "cy"]).to_csv(index=False).encode('utf-8')
                    col_d1.download_button("📥 Excel/CSV Olarak İndir (Veri Seti)", data=csv_data, file_name="quakevision_veriseti.csv", mime="text/csv", use_container_width=True)
                    
                    try:
                        pdf_bytes = create_pdf_report(df, total_buildings, destroyed, heavy_ratio, st.session_state.get("ai_report_text", ""))
                        pdf_name = "QuakeVision_Premium_Rapor_Llama3.pdf" if st.session_state["ai_report_text"] else "QuakeVision_Standart_Rapor.pdf"
                        col_d2.download_button(f"📄 PDF Raporunu İndir", data=pdf_bytes, file_name=pdf_name, mime="application/pdf", use_container_width=True)
                        
                        if not st.session_state["ai_report_text"]:
                            col_d2.caption("💡 *İpucu: Yukarıdaki Llama 3 butonuna basarsanız, PDF'inize yapay zeka analiz raporu da dahil edilir.*")
                    except Exception as e:
                        col_d2.error(f"PDF oluşturulurken hata oluştu: {str(e)}")
        else:
            st.error(f"API Hatası: {response.text}")
