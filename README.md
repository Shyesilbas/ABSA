# Turkish Sentence Sentiment Analysis

Bu proje, bir Türkçe cümleyi okuyup duygu etiketine çevirir:
- `Negative` (olumsuz)
- `Neutral` (nötr)
- `Positive` (olumlu)

Kısa tanım:
- Proje bir yapay zeka modeli eğitir.
- Eğitilen modeli dosyaya kaydeder.
- Yeni cümlelerde tahmin yapar.
- Sonuçları raporlar ve görselleştirir.
- İstenirse web API üzerinden dış sistemlere servis eder.

Bu doküman, yazılım geçmişi olmayan bir kişinin de proje akışını anlayabilmesi için hazırlanmıştır.

---

## 1) Bu Proje Kimin İçin, Ne İşe Yarar?

- **İş birimi / operasyon ekibi:** Müşteri yorumlarını hızlıca olumlu-nötr-olumsuz sınıflandırmak için.
- **Ürün / raporlama ekibi:** Genel duygu dağılımını grafikle izlemek için.
- **Geliştirici ekip:** Modeli yeniden eğitmek, geliştirmek ve API olarak sunmak için.

Gerçek hayatta kullanım örnekleri:
- E-ticaret yorum analizi
- Sosyal medya duygu takibi
- Destek taleplerinin ön sınıflandırılması

---

## 2) Projenin Büyük Resmi (Hiç Teknik Bilmeyen İçin)

Sistemi bir "fabrika hattı" gibi düşünebilirsiniz:

1. **Ham veri gelir**  
   Cümle ve etiketlerden oluşan kayıtlar sisteme alınır.

2. **Veri düzenlenir**  
   Eksik/bozuk satırlar temizlenir, tek tip formata çevrilir.

3. **Model eğitilir**  
   Model, örneklerden öğrenir.

4. **Model doğrulanır**  
   Başarı oranı ölçülür, en iyi model saklanır.

5. **Yeni cümle tahmini yapılır**  
   Model yeni metinlerde duygu etiketi üretir.

6. **Sonuçlar raporlanır**  
   Rapor dosyaları ve grafikler üretilir.

---

## 3) Klasörler Ne İçin Var?

| Klasör / Yol | Ne tutar? | Kim kullanır? |
|---|---|---|
| `data/` | Veri dosyaları (ham, eğitim, test, örnek girişler) | Veri ve model ekipleri |
| `data/outputs/` | Tahmin sonuçları, grafikler, confusion matrix | Analiz ve raporlama ekipleri |
| `models/` | Eğitilen model dosyası | API ve tahmin süreçleri |
| `src/` | Projenin ana Python kodu | Geliştiriciler |
| `backend/` | API servisi | Entegrasyon ekipleri |
| `frontend/` | Basit web arayüzü (varsa) | Demo/kullanıcı tarafı |
| `scripts/` | Yardımcı çalışma dosyaları (Colab vb.) | Geliştiriciler |

---

## 4) "Kim Ne Yapıyor?" — Dosya Bazlı Sorumluluk Haritası

### Çekirdek ayarlar

- `src/config.py`
  - Projenin merkezi ayar dosyasıdır.
  - Veri yolları, model adı, eğitim ayarları burada tutulur.
  - Birçok dosya buradan okuma yapar.

### Veri standardı ve hazırlık

- `src/data_contracts.py`
  - Veri formatının "sözleşmesini" belirler.
  - `Sentence` ve `Polarity` alanlarını normalize eder.
  - Tekrarlayan cümleleri çoğunluk etiketiyle tekilleştirir.

- `src/data_download.py`
  - Ham veriyi dış kaynaktan indirip proje veri klasörüne yazar.

- `src/data_preprocessing.py`
  - Ham veriyi modelin anlayacağı cümle-düzeyi formata dönüştürür.
  - `train/val/test` ayrımı üretir.

- `src/training_data.py`
  - Eğitim havuzunu birleştirir:
    - ana eğitim dosyası
    - opsiyonel ham ABSA dosyası
    - opsiyonel Hugging Face ek veri
    - opsiyonel `hard_examples` baskın etiketleri

- `src/dataset_loader.py`
  - Temizlenmiş veriyi model girişine çevirir.
  - Cümleyi tokenlara ayırır ve tensor üretir.

### Model yükleme, eğitim ve tahmin

- `src/model_loader.py`
  - Model dosyasını güvenli şekilde açar.
  - Dosya yapısını doğrular.
  - Model ve tokenizer'ı çalışmaya hazır döndürür.

- `src/progress.py`
  - Uzun döngülerde terminalde yüzde ve ilerleme çubuğu (`track`, `loader_total`).
  - Eğitim, değerlendirme, batch tahmin ve HF satır işlemede kullanılır.

- `src/trainer.py`
  - Eğitim döngüsünün motorudur.
  - Loss hesabı, epoch eğitim/değerlendirme, checkpoint kaydı ve early stopping burada yönetilir.

- `src/train.py`
  - Orkestrasyon dosyasıdır.
  - "Hangi sırayla ne yapılacak?" kararını verir.
  - Veri hazırlama, model kurma, eğitimi başlatma işlerini birleştirir.

- `src/model_utils.py`
  - Tahmin tarafının yardımcı fonksiyonları.
  - Tek cümle tahmininde etiket ve olasılık üretir.

- `src/predict.py`
  - Tek tek cümle girerek manuel tahmin almak için.

- `src/batch_predict.py`
  - CSV içindeki çok sayıda metni toplu tahmin eder.
  - Sonucu çıktı dosyasına kaydeder.

### Ölçüm ve görselleştirme

- `src/evaluate_metrics.py`
  - Test verisinde model performansını ölçer.
  - Sınıflandırma raporu ve confusion matrix üretir.

- `src/visualize_results.py`
  - Toplu tahmin sonuçlarını grafik haline getirir.

### Servis katmanı

- `backend/main.py`
  - API servisidir.
  - Uygulama açılırken modeli belleğe alır.
  - Sağlık kontrol ve tahmin uçları sunar.

### Yardımcı dosyalar

- `scripts/colab_turkish_sentiment_full.py`
  - Colab ortamı için tek dosyalık eğitim akışı.

- `scripts/inspect_seq_lengths.py`
  - Cümle uzunluklarını analiz etmeye yardımcı olur.

---

## 5) Veri Dosyaları Ne Anlama Geliyor?

| Dosya | Beklenen içerik | Ne için kullanılır? |
|---|---|---|
| `data/turkish_absa_train.csv` | Ham veri (ABSA veya uyumlu format) | Ön işleme / eğitim havuzu |
| `data/train.csv` | `Sentence`, `Polarity` | Model eğitimi |
| `data/val.csv` | `Sentence`, `Polarity` | Eğitim sırasında doğrulama |
| `data/test.csv` | `Sentence`, `Polarity` | Nihai performans ölçümü |
| `data/hard_examples.csv` | Zor örnekler, manuel etiket | Modeli kritik örneklerde güçlendirme |
| `data/sample_tweets.csv` | Toplu tahmin için giriş metinleri | Batch inference |

Notlar:
- `Polarity` değerleri: `0=Negative`, `1=Neutral`, `2=Positive`.
- Aynı cümle birden fazla kez varsa, veri hazırlığında tekilleştirme yapılır.
- `hard_examples` etkinse, çakışan cümlede bu dosyanın etiketi baskın olur.

---

## 6) Çıktılar Nerede ve Ne İfade Ediyor?

| Çıktı | Nerede oluşur? | Anlamı |
|---|---|---|
| Eğitilmiş model | `models/sentence_best_model.bin` | Tahmin yapacak asıl model dosyası |
| Batch tahmin sonucu | `data/outputs/sentiment_batch_results.csv` | Her metin için etiket + güven skoru |
| Confusion matrix | `data/outputs/confusion_matrix.png` | Hangi sınıflar karışıyor görseli |
| Duygu dağılımı grafiği | `data/outputs/chart_sentiment_distribution.png` | Toplam olumlu/nötr/olumsuz sayıları |

---

## 7) Uçtan Uca İş Akışı (Adım Sırası)

### Senaryo A: Sıfırdan model üretmek isteyen ekip

1. Ham veriyi temin et.
   - Dosya: `src/data_download.py`
   - İşlev: `download_and_save_data()`
   - Çıktı: `data/turkish_absa_train.csv`

2. Veriyi standartlaştır ve `train/val/test` üret.
   - Dosya: `src/data_preprocessing.py`
   - İşlevler: `_sentence_level()`, `process_data()`
   - İç çağrı: `src/data_contracts.py` içindeki `prepare_sentence_polarity_frame()` ve `deduplicate_by_sentence_majority()`
   - Çıktılar: `data/train.csv`, `data/val.csv`, `data/test.csv`

3. Eğitim havuzunu birleştir.
   - Dosya: `src/training_data.py`
   - İşlev: `build_train_val_frames()`
   - İç akış:
     - `_prep_csv()` ile veri normalize edilir.
     - `load_hf_subset()` ile opsiyonel HF veri eklenir.
     - `_merge_hard_overrides()` ile opsiyonel hard examples baskınlanır.

4. Model giriş veri setlerini oluştur.
   - Dosya: `src/train.py`
   - İşlev: `build_dataloaders()`
   - Kullanılan sınıf: `src/dataset_loader.py` içindeki `SentenceClassificationDataset`
   - Ne yapar: Cümleleri token/tensor formatına çevirip train/val `DataLoader` üretir.

5. Modeli eğit ve en iyi checkpoint'i kaydet.
   - Dosyalar: `src/train.py`, `src/trainer.py`
   - Orkestrasyon: `train.py > main()`
   - Eğitim motoru:
     - `trainer.build_loss_fn()`
     - `trainer.train_epoch()`
     - `trainer.evaluate_epoch()`
     - `trainer.fit()` (early stopping + checkpoint)
   - Çıktı: `models/sentence_best_model.bin`

6. Test verisiyle kaliteyi ölç.
   - Dosya: `src/evaluate_metrics.py`
   - Akış:
     - Model yükleme: `src/model_loader.py > load_finetuned_resources()`
     - Test dataset: `SentenceClassificationDataset`
     - Tahmin toplama: `collect_predictions()`
     - Rapor/CM: `classification_report`, `plot_cm()`
   - Çıktı: `data/outputs/confusion_matrix.png`

7. Tahmin servislerini çalıştır.
   - Tekli tahmin:
     - Dosya: `src/predict.py`
     - İşlev: `src/model_utils.py > load_classifier(), predict_sentence()`
   - Toplu tahmin:
     - Dosya: `src/batch_predict.py`
     - İşlev: `process_batch()`
     - Çıktı: `data/outputs/sentiment_batch_results.csv`
   - Görselleştirme:
     - Dosya: `src/visualize_results.py`
     - Çıktı: `data/outputs/chart_sentiment_distribution.png`


### Senaryo B: Sadece hazır modelle tahmin yapmak isteyen ekip

1. Model dosyasının mevcut olduğundan emin ol.
2. Modeli yükle (`src/model_utils.py > load_classifier()`).
3. Tekli tahmin için `predict_sentence()` veya toplu tahmin için `src/batch_predict.py > process_batch()` çalıştır.
4. Sonuç CSV ve grafikleri kullan.

---


---

## 9) Sık Karşılaşılan Durumlar ve Anlamları

- **Model dosyası yoksa**
  - Tahmin veya API çalışmaz.
  - Önce eğitim adımı tamamlanmalıdır.

- **Beklenen kolon adı yoksa**
  - Veri okuyucu süreci durdurur.
  - `Sentence` / `Polarity` veya batch tarafında `text` alanı kontrol edilmelidir.

- **Dengesiz sınıf dağılımı varsa**
  - Eğitimde class weight mekanizması devreye girer.
  - Nötr sınıfına ek ağırlık verilebilir.

---

## 10) Ayar Mantığı (Tek Merkezden Yönetim)

Tüm kritik kararlar `src/config.py` içindedir:

- Dosya yolları
- Model adı
- Eğitim hiperparametreleri
- Early stopping ayarları
- Ek veri birleştirme seçenekleri
- Hard example birleştirme davranışı

Bu sayede ekip, farklı dosyalara dağılmadan merkezi konfigürasyonla çalışır.

---

## 10.1) Confidence Fallback Policy (Yeni)

Model düşük güvenle tahmin ürettiğinde, etiketi zorlamamak için fallback politikası uygulanır.

Konfigürasyon (`src/config.py`):
- `CONFIDENCE_FALLBACK_ENABLED`: fallback aktif/pasif.
- `CONFIDENCE_THRESHOLD`: güven eşiği (ör. `0.70`).
- `CONFIDENCE_FALLBACK_LABEL`: eşik altı durumda verilecek etiket (varsayılan `Neutral`).

Davranış:
- Model önce ham tahmini ve olasılıkları üretir.
- `max(probabilities) < CONFIDENCE_THRESHOLD` ise final etiket fallback etikete çevrilir.
- Böylece gri/kararsız Twitter cümlelerinde aşırı güvenli yanlışlar azaltılır.

Etkilenen dosyalar:
- `src/model_utils.py`
  - `predict_sentence()` geriye uyumludur (mevcut kullanım kırılmaz).
  - `predict_sentence_with_meta()` ile `raw_label`, `confidence`, `fallback_applied` bilgileri döner.
- `src/predict.py`
  - Terminal çıktısında fallback uygulandığı görünür.
- `src/batch_predict.py`
  - Çıktı CSV'sine `raw_sentiment` ve `fallback_applied` kolonları eklenir.
  - Süreç sonunda kaç satırda fallback uygulandığı loglanır.

Not:
- Bu politika sadece inference tarafını etkiler, eğitim metriklerini değiştirmez.

---

## 11) Projenin Sınırları

- Bu proje cümle-düzeyi tek etiket duygu analizi yapar.
- ABSA ham verisi kullanılsa da çıktı ABSA seviyesinde değil, cümle seviyesindedir.
- Çok etiketli sınıflandırma veya aspect-level tahmin bu repo kapsamı dışındadır.

---

## 12) Yeni Gelen Bir Kişi Nereden Başlamalı?

Önerilen okuma sırası:

1. `src/config.py`
2. `src/data_contracts.py`
3. `src/training_data.py`
4. `src/train.py`
5. `src/trainer.py`
6. `src/model_loader.py`
7. `src/model_utils.py`
8. `src/evaluate_metrics.py`

Bu sıra, "veri nasıl hazırlanıyor?" sorusundan başlayıp "model nasıl eğitiliyor ve nasıl servis ediliyor?" sorusuna kadar tam resmi verir.
