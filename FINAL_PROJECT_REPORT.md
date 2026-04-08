# **TURKISH SENTENCE-LEVEL SENTIMENT ANALYSIS**

## **(BERTurk Tabanlı Duygu Sınıflandırma Sistemi)**

### **Bitirme Projesi Tam Rapor Metni**

**Öğrenci:** Serhat Haydar Yeşilbaş - Ahmet Emir Solak - Yasin Eren Şahin  
**Üniversite:** Maltepe Üniversitesi  
**Ders/Çalışma Türü:** Kapsamlı Teknik Bitirme Projesi  
**Proje Türü:** Türkçe Cümle Düzeyi Duygu Analizi  
**Model:** `dbmdz/bert-base-turkish-cased` (fine-tuned)  
**Sınıflar:** Negative / Neutral / Positive

---

## **Etik Beyan**

Bu raporda sunulan çalışma, akademik etik kurallarına uygun şekilde yürütülmüştür. Kullanılan yöntemler, veri kaynakları, modelleme tercihleri ve sonuçlar şeffaf biçimde raporlanmıştır.

---

## **Özet**

Bu projede, Türkçe metinler için cümle düzeyinde duygu analizi gerçekleştiren uçtan uca bir sistem geliştirilmiştir. Çalışmanın ana amacı, bir cümleyi `Negative`, `Neutral` veya `Positive` sınıfına yüksek doğrulukla atayabilen, yeniden üretilebilir ve raporlanabilir bir model hattı kurmaktır.

Ana model olarak BERTurk (`dbmdz/bert-base-turkish-cased`) seçilmiş, model fine-tuning ile görev özelinde eğitilmiştir. Veri tarafında standartlaştırma, tekilleştirme, sınıf dengesizliği yönetimi, hard example birleştirme ve isteğe bağlı dış veri genişletme uygulanmıştır. Eğitim sürecinde early stopping, class weights, checkpoint kaydı ve terminal bazlı ilerleme izleme özellikleri kullanılmıştır.

Model performansı yalnızca ana metriklerle değil, baseline karşılaştırmasıyla da doğrulanmıştır. Test sonuçlarında fine-tuned BERT modeli, hem majority baseline hem de TF-IDF + Logistic Regression yaklaşımını belirgin farkla geçmiştir. Ek olarak inference tarafında düşük güvenli tahminler için confidence fallback politikası eklenmiş, belirsiz cümlelerde aşırı güvenli yanlış tahmin riski azaltılmıştır.

**Anahtar Kelimeler:** Türkçe NLP, duygu analizi, BERTurk, metin sınıflandırma, baseline karşılaştırması, confidence fallback

---

## **Abstract**

This project presents an end-to-end sentence-level sentiment analysis system for Turkish text. The goal is to classify each sentence into `Negative`, `Neutral`, or `Positive` with high reliability and reproducibility.

The main model is fine-tuned BERTurk (`dbmdz/bert-base-turkish-cased`). The pipeline includes data standardization, deduplication, class imbalance handling, hard-example merging, and optional external data augmentation. Training uses early stopping, class weighting, checkpointing, and terminal progress monitoring.

The model is evaluated not only by standard metrics but also via baseline comparison. On the test set, the fine-tuned BERT model significantly outperforms both majority-class and TF-IDF + Logistic Regression baselines. Additionally, a confidence fallback policy is integrated for low-confidence predictions to reduce overconfident errors.

**Keywords:** Turkish NLP, sentiment analysis, BERTurk, text classification, baseline comparison, confidence fallback

---

## **1. Giriş**

Dijital platformlarda üretilen kullanıcı metinleri; marka yönetimi, müşteri deneyimi, kriz takibi ve ürün geliştirme açısından kritik içgörü kaynağıdır. Ancak bu metinlerin manuel analizi maliyetli ve ölçeklenemezdir. Duygu analizi bu noktada otomatik sınıflandırma ile karar süreçlerini hızlandırır.

Türkçe, eklemeli yapısı ve bağlam bağımlılığı nedeniyle NLP görevlerinde özel zorluklar barındırır. Bu çalışmada amaç, Türkçe cümle düzeyi duygu analizinde modern transformer yaklaşımı ile yüksek doğruluklu ve pratikte kullanılabilir bir sistem geliştirmektir.

---

## **2. Problem Tanımı ve Kapsam**

### **2.1 Problem**

Verilen bir Türkçe cümlenin duygu etiketini belirlemek:

- `0 -> Negative`
- `1 -> Neutral`
- `2 -> Positive`

### **2.2 Kapsam Dahili**

- Cümle düzeyi tek etiketli sınıflandırma
- Eğitim/validasyon/test ayrımı
- Baseline karşılaştırması
- Batch tahmin ve görsel raporlama
- Güven odaklı inference politikası

### **2.3 Kapsam Dışı**

- Aspect-level inference (ABSA çıktısı üretimi)
- Çok etiketli sınıflandırma
- Çok dilli genelleme

> *Not: Proje ilk aşamada ABSA odaklı tasarlanmış olsa da güncel sürüm cümle düzeyi sentiment sınıflandırmaya evrilmiştir. Mimari buna göre sadeleştirilmiştir.*

---

## **3. Literatür Özeti**

Duygu analizi literatüründe klasik yöntemler (Bag-of-Words, TF-IDF, SVM, Naive Bayes) düşük maliyetli ve hızlıdır; fakat bağlamı sınırlı temsil eder. Derin öğrenme ile birlikte RNN/LSTM tabanlı çözümler bağlamı daha iyi modellemiştir. Transformer mimarisi ve özellikle BERT ailesi, self-attention mekanizması ile bağlamsal temsilde yeni bir standart oluşturmuştur.

Türkçe özelinde BERTurk modeli, dilin morfolojik yapısına uygun ön eğitim avantajı sunar. Bu nedenle proje, ana omurga olarak BERTurk’u kullanmıştır.

---

## **4. Veri, Ön İşleme ve Veri Sözleşmesi**

### **4.1 Veri Kaynakları**

- Yerel `train.csv`, `val.csv`, `test.csv`
- Opsiyonel ham ABSA kaynak dosyası
- Opsiyonel Hugging Face ek eğitim verisi
- Opsiyonel `hard_examples.csv` (zor örnekler)

### **4.2 Veri Sözleşmesi**

Modelin standart beklediği kolonlar:

- `Sentence` (metin)
- `Polarity` (etiket: 0/1/2)

### **4.3 Ön İşleme Adımları**

- Kolon adı normalize etme (`text -> Sentence`, `label -> Polarity`)
- `Aspect` kolonu varsa kaldırma
- Boş/NaN kayıtları temizleme
- Metin trim işlemleri
- Cümle bazlı tekilleştirme (çoğunluk etiketi)
- Hard examples çakışmasında manuel etiketin baskın olması

### **4.4 Veri Bölünmesi**

Veri train/val/test olarak ayrılmıştır. Test seti eğitimde kullanılmamıştır.

---

## **5. Sistem Mimarisi ve Uygulama Bileşenleri**

### **5.1 Genel Mimari**

Sistem modüler bir katman yapısıyla tasarlanmıştır:

1. **Veri Hazırlama Katmanı**
2. **Model Eğitim Katmanı**
3. **Model Yükleme & Inference Katmanı**
4. **Değerlendirme & Raporlama Katmanı**

### **5.2 Ana Dosyalar ve Görevleri**

- `config.py`: Merkezi ayarlar
- `data_contracts.py`: Veri sözleşmesi/normalize/tekilleştirme
- `training_data.py`: Eğitim havuzu birleştirme
- `dataset_loader.py`: Tokenize/tensor dataset üretimi
- `train.py`: Eğitim orkestrasyonu
- `trainer.py`: Epoch eğitim/değerlendirme motoru
- `model_loader.py`: Checkpoint güvenli yükleme
- `model_utils.py`: Tahmin fonksiyonları
- `evaluate_metrics.py`: Test metrikleri + confusion matrix
- `batch_predict.py`: Toplu tahmin
- `progress.py`: Terminal yüzde/progress altyapısı
- `baseline_eval.py`: Baseline karşılaştırma
- `ablation_plan.py`: Ablation plan matrisi üretimi

---

## **6. Modelleme Yaklaşımı**

### **6.1 Ana Model**

- Model: `dbmdz/bert-base-turkish-cased`
- Görev: Sequence classification (`num_labels=3`)

### **6.2 Eğitim Stratejisi**

- Optimizer: AdamW
- Öğrenme oranı: `2e-5`
- Epoch: `4`
- Batch size: `32`
- Max length: `160`
- Warmup ratio: `0.1`
- Early stopping: açık
- Sınıf ağırlıkları: açık
- Neutral class loss boost: `1.0` (güncel koşu)
- Checkpoint: en iyi validasyon sonucu kaydedilir

### **6.3 Inference Güvenlik Politikası**

- Confidence fallback aktif
- Düşük güvenli tahminlerde etiket `Neutral`a yönlendirilebilir
- Batch sonuçlarında:
  - `sentiment` (nihai)
  - `raw_sentiment` (fallback öncesi)
  - `fallback_applied` (True/False)

---

## **7. Deneysel Kurulum**

### **7.1 Karşılaştırılan Modeller**

1. `majority_class`
2. `tfidf_logreg`
3. `bert_finetuned`

### **7.2 Değerlendirme Metrikleri**

- Accuracy
- Macro-F1
- Sınıf bazlı precision/recall/F1
- Confusion matrix (ana model)

### **7.3 Test Verisi**

Toplam test örneği: **2050**

---

## **8. Sonuçlar (Güncel Run: `run_default`)**

### **8.1 Genel Karşılaştırma Sonuçları**

| **Model** | **Accuracy** | **Macro-F1** |
| --- | ---: | ---: |
| **bert_finetuned** | **0.973659** | **0.961286** |
| tfidf_logreg | 0.877073 | 0.796227 |
| majority_class | 0.520488 | 0.228211 |

### **8.2 Sınıf Bazlı Sonuçlar (Güncel Evaluate)**

- Negative: P=0.9821, R=0.9660, F1=0.9740
- Neutral: P=0.9242, R=0.9385, F1=0.9313
- Positive: P=0.9731, R=0.9841, F1=0.9786

### **8.3 Confusion Matrix Yorumları (BERT, run_default)**

Sayısal tablo:

- Negative (gerçek): 824 doğru, 5 neutral, 24 positive
- Neutral (gerçek): 122 doğru, 3 negative, 5 positive
- Positive (gerçek): 1050 doğru, 12 negative, 5 neutral

Özet:

- Sınıflar arası ayrım güçlü
- En doğal hata bölgesi `Negative <-> Positive` sınır cümleleri
- Neutral performansı güçlü seviyededir

### **8.4 Hata Analizi Çıktıları**

- `data/outputs/run_default/test_misclassified.csv`
- `data/outputs/run_default/test_confusion_pairs.csv`

Bu dosyalar hangi cümlelerde ve hangi sınıf çiftlerinde hata yoğunlaştığını açıkça gösterir.

### **8.5 Batch Inference Nitel Analizi (`sentiment_batch_results.csv`)**

`data/outputs/sentiment_batch_results.csv` üzerinde 30 örneklik kalite kontrolü yapılmıştır.

Öne çıkan bulgular:

- **Fallback davranışı aktif ve işlevsel:** 30 örneğin 3 tanesinde (`id=9,17,20`) `fallback_applied=True` görülmüştür.
- Bu üç örnekte modelin ham tahmini (`raw_sentiment`) düşük güvenle gelmiş, final etiket güvenlik politikası ile `Neutral` olarak güncellenmiştir.
- **Düşük güven eşiği altında düzeltme örnekleri:** `confidence=0.5475`, `0.6321`, `0.4370`.
- Net kutuplu cümlelerde model yüksek güvenle tutarlı tahmin üretmiştir (çoğunlukla `confidence > 0.95`).
- Kararsız/nötre yakın bazı ifadelerde (`id=10`, `id=30`) negatif tarafa eğilim devam etmektedir; bu örnekler gelecekte `hard_examples.csv` ile güçlendirme adayıdır.

Yorum:

Bu sonuçlar, confidence fallback katmanının pratikte aşırı güvenli yanlışları azaltarak daha temkinli ve raporlanabilir bir batch davranışı sağladığını göstermektedir.

---

## **9. Tartışma**

### **9.1 Teknik Yorum**

BERT modeli klasik baseline’lara göre belirgin üstünlük sağlamıştır. Bu fark özellikle bağlam gerektiren cümlelerde ve neutral sınıfında netleşmektedir. TF-IDF yaklaşımı güçlü bir referans olmasına rağmen bağlam bilgisinin sınırlı olması nedeniyle transformer tabanlı modelin gerisinde kalmıştır.

### **9.2 Riskler ve Sınırlılıklar**

- İroni/sarkazm cümleleri halen zorlu
- Neutral sınıfı doğası gereği belirsiz
- Domain kayması durumunda yeniden fine-tuning gerekebilir
- Çok kısa ve bağlamı eksik cümlelerde model güveni yapay yüksek olabilir (fallback bu riski azaltır)

---

## **10. Endüstriyel Uygulanabilirlik**

Sistem aşağıdaki alanlara doğrudan uyarlanabilir:

- E-ticaret yorum analizi
- Sosyal medya duygu takibi
- Müşteri destek metinlerinin ön sınıflandırılması
- Kriz iletişimi ve marka algısı izleme

---

## **11. Sonuç ve Gelecek Çalışmalar**

Bu çalışma, Türkçe cümle düzeyi duygu analizi için modüler, ölçülebilir ve güçlü bir teknik altyapı ortaya koymuştur. Ana model, baseline’lara göre istatistiksel olarak anlamlı performans farkı üretmiştir.

Gelecek adımlar:

1. Ablation deneyleri (`USE_HF_TRAIN_EXTRA`, `MERGE_HARD_EXAMPLES`, fallback açık/kapalı)
2. Sistematik hata analizi raporu (yanlış örnek kümeleri)
3. Deney artifact kayıt standardını koşu bazlı sürümlemek
4. Domain-spesifik veri artırımı ve yeniden eğitim

---

## **Kaynakça**

1. Devlin, J. et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
2. Vaswani, A. et al. (2017). Attention Is All You Need.
3. Schweter, S. (2020). BERTurk: Turkish BERT models.
4. Liu, B. (2012). Sentiment Analysis and Opinion Mining.
5. Pang, B. & Lee, L. (2008). Opinion Mining and Sentiment Analysis.
6. scikit-learn Documentation.
7. Hugging Face Transformers Documentation.

---

## **Ekler**

### **Ek-A: Üretilen Çıktılar (run_default)**

- `data/outputs/run_default/baseline_comparison.csv`
- `data/outputs/run_default/baseline_class_reports.csv`
- `data/outputs/run_default/confusion_matrix.png`
- `data/outputs/run_default/test_misclassified.csv`
- `data/outputs/run_default/test_confusion_pairs.csv`

### **Ek-B: Deney Artifact**

- `data/outputs/experiment_last_run.json`

### **Ek-C: Kullanılan Temel Konfigürasyon Özeti**

- Model: BERTurk
- Epoch: 4
- Batch Size: 32
- Learning Rate: 2e-5
- Max Length: 160
- Early Stopping: Açık
- Class Weights: Açık
- Neutral Loss Boost: 1.0
- Confidence Fallback: Açık

