from pydantic import BaseModel

class ReportRequest(BaseModel):
    total_buildings: int
    destroyed: int
    heavy_damage: int
    light_damage: int
    intact: int
    model_name: str = "llama3.2"
    api_key: str = None

def generate_disaster_report(data: ReportRequest) -> str:
    try:
        import datetime
        tarih = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
        hasar_orani = 0
        if data.total_buildings > 0:
            hasar_orani = ((data.destroyed + data.heavy_damage) / data.total_buildings) * 100
            
        prompt = f"""Sen uzman bir AFAD ve GIS (Coğrafi Bilgi Sistemleri) afet hasar tespit mühendisisin. 
Aşağıdaki QuakeVision yapay zeka uydu analizi sonuçlarına göre resmi bir "Bölge Durum ve Acil Müdahale Raporu" oluştur.

RAPOR VERİLERİ:
- Rapor Tarihi: {tarih}
- Toplam İncelenen Bina: {data.total_buildings}
- Yıkılmış Bina: {data.destroyed}
- Ağır Hasarlı Bina: {data.heavy_damage}
- Hafif Hasarlı Bina: {data.light_damage}
- Sağlam Bina: {data.intact}
- Ağır Hasar Oranı: %{hasar_orani:.1f}

Lütfen raporunu çok detaylı, kapsamlı ve derinlemesine bir analiz sunacak şekilde, aşağıdaki başlıklara ayırarak son derece resmi, kurumsal ve akademik bir Türkçe ile yaz (Markdown formatında):

## 1. DURUM ÖZETİ VE KRİZİN BOYUTU
Mevcut durumun aciliyetini, genel tabloyu ve bölgesel etkinin çapını detaylıca özetle. Ortaya çıkan hasar tablosunun bölge halkı ve altyapı üzerindeki olası etkilerini stratejik bir bakış açısıyla yorumla.

## 2. DETAYLI HASAR VE RİSK ANALİZİ
Verilen rakamları ve özellikle ağır hasar oranını baz alarak tehlikenin boyutunu derinlemesine analiz et. Yıkılan ve ağır hasarlı binaların etrafındaki sağlam yapılara verebileceği zincirleme zararları, olası artçı sarsıntılardaki göçme risklerini ve bölgedeki altyapı (yol, su, elektrik) çökme ihtimallerini değerlendir.

## 3. ARAMA KURTARMA (SAR) VE SAHA OPERASYONLARI
Yıkılmış ({data.destroyed}) bina sayısına göre net ve matematiksel tahminler yürüt:
- Kaç adet tam donanımlı SAR (Arama Kurtarma) ekibine ihtiyaç var?
- Hangi tip ağır iş makineleri (vinç, ekskavatör) ve sismik/akustik dinleme cihazları bölgeye sevk edilmeli?
- Enkaz altındaki olası yaralı sayısına göre ne kadarlık bir triyaj alanı ve sahra hastanesi kurulmalı?

## 4. LOJİSTİK, TAHLİYE VE GEÇİCİ BARINMA (ÇADIRKENT) PLANLAMASI
Evsiz kalan vatandaşlar için çok detaylı bir lojistik planı hazırla:
- Günlük kaç ton temiz su ve gıda ihtiyacı doğacaktır?
- Çadırkent veya konteyner kent kurulumu için ne kadarlık bir alan ve jeneratör altyapısı gerekir?
- Bölgedeki salgın hastalık riskini önlemek için mobil tuvalet/duş ve hijyen kitleri dağıtımı nasıl koordine edilmeli?

Kurallar:
- Çıktın çok detaylı ve uzun olmalı. Her başlık altında en az 2-3 paragraf açıklama ve maddeler halinde stratejik öneriler bulunmalı.
- Sadece raporu yaz, ekstra giriş (merhaba, işte rapor vb.) veya çıkış cümlesi kurma.
- Rapor sanki İçişleri Bakanına veya Afet Yönetim Merkezi (AFAD) Başkanına sunulacakmış gibi aşırı ciddi, otoriter ve veri odaklı bir tonda olsun.
- Alt başlıkları (###) ve vurguları (**) profesyonelce kullan.
"""
        
        if data.api_key and data.api_key.strip() != "":
            import google.generativeai as genai
            genai.configure(api_key=data.api_key.strip())
            model = genai.GenerativeModel('gemini-3.5-flash')
            response = model.generate_content(prompt)
            return response.text
        else:
            from langchain_ollama import OllamaLLM
            llm = OllamaLLM(model=data.model_name, base_url="http://localhost:11434")
            response = llm.invoke(prompt)
            return response
            
    except Exception as e:
        return f"Rapor oluşturulurken hata meydana geldi.\n\nSistem Hatası: {str(e)}"
