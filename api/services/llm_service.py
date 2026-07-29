from pydantic import BaseModel

class ReportRequest(BaseModel):
    total_buildings: int
    destroyed: int
    heavy_damage: int
    light_damage: int
    intact: int
    model_name: str = "llama3"

def generate_disaster_report(data: ReportRequest) -> str:
    try:
        from langchain_community.llms import Ollama
        
        # Initialize Ollama LLM
        llm = Ollama(model=data.model_name, base_url="http://localhost:11434")
        
        prompt = f"""Sen uzman bir AFAD ve GIS (Cografi Bilgi Sistemleri) afet hasar tespit muhendisisin. 
Asagidaki QuakeVision yapay zeka uydu analizi sonuclarina gore resmi bir "Bolge Durum ve Acil Mudahale Raporu" olustur.

Veriler:
- Toplam Bina Sayisi: {data.total_buildings}
- Yikilmis Bina: {data.destroyed}
- Agir Hasarli Bina: {data.heavy_damage}
- Hafif Hasarli Bina: {data.light_damage}
- Saglam Bina: {data.intact}

Lutfen raporunu su basliklara ayirarak resmi bir Turkce dille yaz (markdown formatinda):
1. Mevcut Durum Ozeti
2. Hasar Analizi ve Kritik Oranlar
3. Arama Kurtarma (SAR) ve Lojistik Onerileri

Kurallar:
- Sadece raporu yaz, ekstra giris (merhaba vs.) veya cikis cumlesi kurma.
- Rakamlari ve oranlari analiz ederek tehlikenin boyutunu yorumla.
"""
        
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Rapor olusturulurken hata meydana geldi. Bilgisayarinizda Ollama'nin calistigindan ve '{data.model_name}' modelinin yuklu oldugundan emin olun.\n\nSistem Hatasi: {str(e)}"
