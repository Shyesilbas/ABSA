# 1. AWS mimarisine (x86_64) uygun Python sürümünü seçiyoruz (M3 uyumsuzluğunu çözer)
FROM --platform=linux/amd64 python:3.10-slim

# 2. Konteyner içindeki çalışma klasörümüzü belirliyoruz
WORKDIR /app

# 3. Sistem bağımlılıklarını güncelliyoruz
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Sadece requirements dosyasını kopyalayıp kütüphaneleri kuruyoruz
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Hugging Face modelinin her seferinde internetten inmemesi için önbellek klasörünü çalışma dizinine ayarlıyoruz
ENV HF_HOME=/app/.cache/huggingface

# 6. Projedeki tüm dosyaları (modeller dahil) Docker içine kopyalıyoruz
COPY . .

# 7. FastAPI'nin çalışacağı portu açıyoruz
EXPOSE 8000

# 8. Uygulamayı ayağa kaldıran komut (ekran görüntündeki backend klasörüne göre ayarlandı)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]