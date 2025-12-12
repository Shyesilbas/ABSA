# Turkish Aspect-Based Sentiment Analysis (ABSA) Projesi

Bu proje, Türkçe metinlerde (özellikle sosyal medya verisi) **Aspect-Based Sentiment Analysis (ABSA)** gerçekleştirmek için geliştirilmiş BERT tabanlı bir NLP sistemidir.

---

##  Başlangıç Rehberi

Kodları çalıştırmadan önce aşağıdaki kurulum adımlarını tamamlayın.

Kodlarda import etmeniz gereken kısımlar kırmızı gözükecetir. Onları import edin.

### 1. Klasörleri Oluşturun
Projeyi indirdikten sonra kök dizinde şu iki klasörün olduğundan emin olun (yoksa sağ tıklayıp oluşturun):

- **`data/`**: Veri setlerinin ve analiz sonuçlarının tutulacağı yer.
  - *Not:* `data_download.py` çalışınca `turkish_absa_train.csv` buraya otomatik gelecek.
  - *Not:* `data_preprocessing.py` çalışınca `train.csv`, `val.csv`, `test.csv` buraya otomatik gelecek.
  
- **`models/`**: Eğitilmiş model dosyasının koyulacağı yer.
  - Elinizdeki **`best_model_state.bin`** dosyasını bu klasörün içine yapıştırın.



### 2. Türkçe Dil Modelini (SpaCy) İndirin
Otomatik özellik çıkarımı (Aspect Extraction) için SpaCy gereklidir. Terminalde sırasıyla şunları deneyin:

```bash
# Yöntem 1 (Önerilen)
python -m spacy download tr_core_news_tr
```

*Hata alırsanız:*
```bash
# Yöntem 2
python -m spacy download tr_core_news_md
```

*Hala hata alıyorsanız (Alternatif):*
```bash
pip install https://huggingface.co/turkish-nlp-suite/tr_core_news_md/resolve/main/tr_core_news_md-1.0-py3-none-any.whl
```

---

## ️ Adım Adım Çalıştırma

Sistemi uçtan uca çalıştırmak için `src/` klasöründeki dosyaları aşağıdaki sırayla çalıştırın.

### Adım 1: Veriyi İndirme 
HuggingFace üzerinden ham veri setini indirmek için:
*   Çalıştır: **`src/data_download.py`**
*   *Sonuç:* `data/turkish_absa_train.csv` dosyası oluşur.

### Adım 2: Veriyi İşleme ve Bölme 
Veriyi temizlemek ve Eğitim/Test olarak ayırmak için:
*   Çalıştır: **`src/data_preprocessing.py`**
*   *Sonuç:* `data/` klasöründe `train.csv`, `val.csv` ve `test.csv` dosyaları oluşur.

### Adım 3: Model Eğitimi ️ (Opsiyonel)
Sıfırdan model eğitmek veya mevcut modeli tazelemek isterseniz:
*   Çalıştır: **`src/train.py`**
*   *Ne yapar?* `train.csv` verisiyle BERT modelini eğitir ve en iyi sonucu `models/best_model_state.bin` olarak kaydeder.
*(Not: Eğitim işlemi CPU üzerinde çok yavaş olabilir. Mümkünse GPU kullanılması önerilir.)*

### Adım 4: Tahmin ve Analiz (Prediction) 🔮

Projede üç farklı tahmin yöntemi vardır:

| Dosya | Açıklama                                                                                                                           |
| :--- |:-----------------------------------------------------------------------------------------------------------------------------------|
| **`auto_predict.py`** | **Tam Otomatik.** Sadece cümleyi girersiniz, model hem özelliği (aspect) bulur hem de duygu analizi yapar.                         |
| **`predict.py`** | **Manuel.** Cümleyi ve analiz edilecek özelliği (aspect) sizin girmeniz gerekir.                                                   |
| **`batch_predict.py`** | **Toplu Analiz.** `data/sample_tweets.csv` dosyasındaki binlerce satırı tek seferde analiz eder. Colabden yapmanız tavsiye edilir. |

**Öneri:** Hızlı sonuç görmek için `auto_predict.py` çalıştırın.
*(Not: Çok büyük verilerle `batch_predict.py` çalıştıracaksanız, hız için kodu Google Colab'e taşıyıp T4 GPU seçerek çalıştırmanız önerilir. Çıkan `final_report.csv` dosyasını tekrar `data/` klasörüne atabilirsiniz.)*

### Adım 5: Model Performansını Ölçme (Metrikler) 📈
Modelin doğruluk oranını (Accuracy), F1-Score ve Confusion Matrix değerlerini görmek için:
*   Çalıştır: **`src/evaulate_metrics.py`**
    *   *Ne yapar?* Test veri setini (`test.csv`) kullanarak modelin başarısını sayısal olarak ölçer ve raporlar.

### Adım 6: Sonuçları Görselleştirme 
Çıkan analiz sonuçlarını (final_report.csv) grafiğe dökmek için:
*   Çalıştır: **`src/visualize_results.py`**
*   *Sonuç:* `data/` klasörüne `.png` formatında grafikler kaydedilir.

---

##  Özet: Sıfırdan Çalıştırma Sırası (Pipeline)

Geliştirme sürecini baştan sona test etmek istiyorsanız, dosyaları şu sırayla çalıştırın:

1.  **`src/data_download.py`** ➔ Veriyi indirir.
2.  **`src/data_preprocessing.py`** ➔ Veriyi temizler ve böler.
3.  **`src/train.py`** ➔ Modeli eğitir (Opsiyonel).
4.  **`src/auto_predict.py`** ➔ Otomatik tahmin yapar.
5.  **`src/predict.py`** ➔ Manuel tahmin yapar.
6.  **`src/batch_predict.py`** ➔ Toplu analiz yapar.
7.  **`src/evaluate_metrics.py`** ➔ Başarı ölçümü yapar.
8.  **`src/visualize_results.py`** ➔ Sonuçları grafikleştirir.

---

##  Karşılaşabileceğiniz Hatalar


**Hata:** `FileNotFoundError: Model file not found...`
*   **Çözüm:** `models/best_model_state.bin` dosyasının doğru klasörde olduğundan emin olun.

**Hata:** `OSError: [E050] Can't find model 'tr_core_news_tr'`
*   **Çözüm:** Yukarıdaki "3. Türkçe Dil Modelini İndirin" başlığındaki komutları deneyin.

---
