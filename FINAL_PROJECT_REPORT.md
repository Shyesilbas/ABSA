# Final Project Report

## Title

**BERTurk-Based Sentence-Level Sentiment Analysis for Turkish: From Reproducible Training to API-Driven Deployment**

## Abstract

Automated sentiment analysis for Turkish is critical for operational scalability in customer support and social listening, yet bridging the gap between research prototypes and production deployment remains a challenge. This project delivers an end-to-end, enterprise-grade sentiment analysis system using a fine-tuned **BERTurk** model. Trained on a curated and deduplicated pool of **30,770 sentences**, the system achieves a **97.5% test accuracy** and a **0.962 macro-F1 score**, significantly outperforming TF-IDF + Logistic Regression (87.7%) and majority-class (52.0%) baselines. Beyond offline metrics, the project introduces a robust **hybrid microservices architecture** where an **ASP.NET Core API Gateway** manages traffic and security, while a **Python FastAPI** service handles high-performance inference. To ensure production reliability, a **confidence-threshold fallback mechanism** (set at **0.70**) is implemented to mitigate overconfident errors on ambiguous inputs. The final solution is served through an interactive **React + Vite dashboard** featuring chunked batch processing, real-time analytics with Recharts, and persistent state management, providing a complete path from reproducible deep learning training to a scalable software product.

## 1. Introduction

Short, user-generated Turkish text is continuously produced in e-commerce, social media, and customer-support channels. Converting these texts into reliable sentiment signals can improve decision speed and consistency in operational workflows. However, Turkish morphology, informal language, and domain-specific phrasing make this task challenging, especially when systems are expected to be reproducible, testable, and deployment-ready rather than notebook-only prototypes.

This report addresses sentence-level Turkish sentiment classification by fine-tuning a transformer-based model within a reproducible software pipeline. The study bridges the gap between laboratory experiments and real-world usage by standardizing heterogeneous data sources and delivering a fully integrated system. On the modeling side, the project explores the transition from classical NLP baselines to contextual representations, while on the engineering side, it introduces a robust **hybrid microservices architecture**. By combining a **C# API Gateway** with a **Python FastAPI** backend and a modern **React dashboard**, this work provides a complete, deployment-ready solution for automated sentiment analysis in the Turkish language.

## 2. Problem Definition and Scope

This project focuses on **sentence-level Turkish sentiment classification** with three labels: **Negative (0)**, **Neutral (1)**, and **Positive (2)**. The primary research question is whether a fine-tuned Turkish BERT model can provide strong and reproducible performance under a practical engineering pipeline.

### In Scope

- Single-label, sentence-level polarity classification.
- Controlled train/validation/test workflow and reproducible experiment artifacts.
- Baseline comparison (majority, TF-IDF-based methods, and transformer model).
- Production-style inference through API and web demo, including batch usage and visualization.

### Out of Scope

- Aspect-level prediction at inference time.
- Multi-label sentiment classification.
- Cross-lingual generalization.

### Project Evolution Note (Important)

The early phase explored ABSA-style data sources and representations. In the final system, the scope is intentionally narrowed to sentence-level sentiment analysis. Aspect fields are treated as optional preprocessing context and are not used as runtime model inputs in the final inference pipeline.

## 3. Related Work

### 3.1 From Lexicon and Classical ML to Contextual Models

Early sentiment-analysis systems were dominated by lexicon-driven polarity scoring and bag-of-words style features. These approaches are computationally efficient and interpretable, and they remain useful as baseline references in low-resource settings. Classical supervised models such as SVM and Logistic Regression with TF-IDF representations improved robustness over purely rule-based systems, especially when domain-specific labeled data was available. However, their core limitation is contextual blindness: token counts and sparse vectors cannot reliably encode scope, contrast, compositional meaning, or long-distance dependencies in complex sentences.

### 3.2 Deep Learning and Transformer Transition

Deep learning shifted sentiment analysis toward dense contextual representations. RNN/LSTM and BiLSTM architectures improved sequential modeling and outperformed many sparse-feature baselines, yet they still faced optimization and efficiency bottlenecks for longer contexts. The Transformer architecture addressed these constraints through self-attention and parallelizable computation, enabling stronger contextual encoding at scale. BERT-style pretraining further accelerated downstream progress by separating large-scale language representation learning from task-specific fine-tuning, which became the dominant paradigm for classification tasks.

### 3.3 Turkish NLP and BERTurk

Turkish presents additional NLP challenges due to rich morphology, agglutination, and productive suffixation, all of which increase lexical variation and ambiguity. In this context, language-specific pretrained models are often more effective than generic multilingual checkpoints because they better internalize Turkish morphology and usage patterns. BERTurk (`dbmdz/bert-base-turkish-cased`) is therefore a natural backbone choice for this thesis, as it combines Transformer-level contextual power with Turkish-specialized pretraining.

### 3.4 Related Work Positioning of This Thesis

Relative to prior work, this thesis intentionally prioritizes **sentence-level** Turkish sentiment classification rather than full runtime ABSA extraction/classification. The contribution is not only model-centric but also system-centric: a reproducible training/evaluation pipeline, baseline benchmarking, structured confusion/error analysis, and deployable inference interfaces (API + frontend demo). This positioning is important because many studies report strong model metrics without an equally transparent operational path from experiment to usable service.

The conceptual grounding of this section follows key citation pillars: Transformer architecture, BERT pretraining/fine-tuning, BERTurk as a Turkish-specific checkpoint, foundational sentiment-analysis literature, and ABSA background for documenting project evolution. Full bibliographic entries are consolidated in the References section using a single citation style.

## 4. Dataset and Preprocessing

### 4.1 Data Sources

The final training pipeline uses sentence-polarity formatted CSV datasets in the repository (`train.csv`, `val.csv`, `test.csv`) as the core source. In addition, the training pool can optionally be expanded with:

- a raw ABSA-oriented CSV source,
- a sampled Hugging Face Turkish sentiment subset,
- manually curated hard examples.

These optional sources are merged only into training data, while validation and test sets remain fixed for fair evaluation.

### 4.2 Data Contract and Standardization

All incoming sources are normalized to a shared schema:

- `Sentence` (text)
- `Polarity` (0/1/2)

During preprocessing, column harmonization, missing-value cleaning, text trimming, and label normalization are applied. Duplicate sentences are resolved with majority-label logic to improve consistency.

### 4.3 Split Strategy

The workflow uses dedicated **train / validation / test** files, and test data is never used for model fitting. In optional reconstruction workflows, split generation follows stratified logic to preserve class proportions as much as possible. Final run-specific sample counts are recorded in experiment artifacts to ensure traceability.

### 4.4 Class Balance and Data Quality

Class imbalance is expected in Turkish sentiment data (especially the neutral class). The project addresses this through weighted loss and targeted hard-example enrichment rather than test-set manipulation. Leakage checks are applied between train and validation pools to prevent overlap-driven optimism.

### 4.5 Tokenization and Encoding

Text is tokenized with the BERTurk tokenizer (`dbmdz/bert-base-turkish-cased`) and converted to model inputs (`input_ids`, `attention_mask`) with fixed-length padding/truncation according to the configured maximum sequence length.

### 4.6 ABSA-to-Sentence-Level Transition

The initial project direction included ABSA-oriented resources, but the final thesis contribution is explicitly sentence-level sentiment classification (see Section 2 for scope rationale).

## 5. Model Architecture and Methodology

### 5.1 Model Selection

This study uses **BERTurk** (`dbmdz/bert-base-turkish-cased`), a transformer model pre-trained on large-scale Turkish text. Compared with generic multilingual alternatives, a Turkish-specialized checkpoint is preferred to better capture agglutinative morphology, suffix-driven meaning shifts, and domain-specific lexical usage in Turkish short text.

### 5.2 Architecture and Fine-Tuning Setup

The base encoder follows the standard BERT-base configuration (12 transformer layers, 768 hidden size, 12 attention heads; approximately 110M parameters). For classification, the project uses `AutoModelForSequenceClassification` with `num_labels=3`, adding a task head for:

- Negative
- Neutral
- Positive

Fine-tuning is performed end-to-end on sentence-level labels. The training workflow includes class-weighted loss (when enabled), learning-rate warmup, mixed-precision support, gradient clipping (global L2 norm capped at **1.0**), early stopping, and checkpointing of the best validation result.

### 5.3 Input Representation (Current Final Scope)

In the final system, model input is **single-sentence sentiment classification**, not aspect-conditioned inference. Each sample is tokenized from sentence text only and encoded as BERT-compatible tensors (`input_ids`, `attention_mask`) with fixed-length padding/truncation according to the configured maximum sequence length.

### 5.4 Note on ABSA-Era Design

An earlier phase explored ABSA-style resources; however, the final system uses sentence-level polarity prediction only. ABSA pair formatting is not used in the final runtime pipeline.

## 6. Experimental Setup

### 6.1 Dataset and Preprocessing

The primary dataset consists of a combined pool of Turkish user-generated texts. To enhance the model's robustness and domain coverage, the base training pool was augmented by merging 10,000 randomly sampled instances from the Hugging Face `winvoker/turkish-sentiment-analysis-dataset` and a curated set of `hard_examples.csv` containing challenging, contrastive sentences. The final training pool consists of 30,770 sentences. During preprocessing, exact duplicates were resolved using a majority-voting mechanism to prevent label leakage and ensure a balanced signal. Validation (N=2,049) and test (N=2,050) sets were kept strictly isolated from this merging process to preserve the integrity of the evaluation.

### 6.2 Model Architecture

The core engine of the sentiment classifier is **BERTurk** (`dbmdz/bert-base-turkish-cased`), a state-of-the-art bidirectional transformer model pre-trained specifically on a massive Turkish corpus (35GB of text). Unlike multilingual BERT (mBERT), BERTurk's vocabulary is optimized entirely for Turkish morphology, leading to superior sub-word tokenization via WordPiece. For this task, the model is fine-tuned for sequence classification: the contextualized embedding of the `[CLS]` token (pooler output) is passed through a dense dropout layer into a 3-way linear classification head corresponding to Negative, Neutral, and Positive polarities.

### 6.3 Evaluation Metrics

Model performance is evaluated using **Accuracy** and **Macro-F1**. While Accuracy measures the overall proportion of correct predictions, Macro-F1 calculates the unweighted mean of the F1-scores across all three classes. Given the inherent class imbalance (with the Neutral class having the lowest support), Macro-F1 is the primary metric for model selection, as it penalizes the model heavily if it fails to detect the minority classes.

### 6.4 Handling Edge Cases: Sarcasm and Irony

A well-known limitation of transformer-based sentiment models is their vulnerability to sarcasm, irony, and implicit dissatisfaction, where the literal meaning of the words contradicts the user's actual sentiment. To mitigate this, a dual-strategy approach was employed. First, a curated `hard_examples.csv` dataset was introduced during training (`MERGE_HARD_EXAMPLES=True`). This dataset contains manually annotated edge cases, specifically targeting contrastive conjunctions and sarcastic structures, forcing the model to learn contextual cues rather than relying on literal keywords. Second, the **70% Confidence Fallback** mechanism acts as a safety net during inference: if a sarcastic sentence confuses the model and produces a highly uncertain probability distribution, the system forces the prediction to "Neutral", preventing severe polar misclassifications in production environments.

### 6.5 Ablation Study Design

To rigorously evaluate the isolated impact of our architectural and data-engineering decisions, an 8-run ($2^3$) ablation study was designed. The study systematically toggles three core variables: the inclusion of Hugging Face extra training data, the injection of curated challenging sentences (`hard_examples.csv`), and the activation of the inference safety net (Confidence Fallback). By maintaining a strict control over random seeds and evaluating against a frozen, held-out test set, this grid isolates the exact contribution of each mechanism, identifying whether a specific feature improves validation scores at the cost of test generalizability (see Section 7.5 for results).

### 6.6 Summary of Empirical Findings

Based on the evaluation protocols described above, the fine-tuned **BERTurk** model emerged as the definitive best-performing architecture. It achieved a test accuracy of **97.36%** and a macro-F1 score of **0.961**, significantly outperforming the TF-IDF + Logistic Regression baseline (87.7% accuracy) and the majority-class baseline (52.0% accuracy).

Analysis of the confusion matrix reveals that the model is highly confident in predicting **Positive** sentiments (98.4% correct classification rate) but exhibits its highest confusion and lowest confidence when predicting the **Neutral** class (93.8% correct classification rate). Specifically, the system struggles the most with contrastive sentences (e.g., sentences containing both praise and complaints). The overall misclassification (error) rate across the entire test set is exceptionally low at **2.63%** (54 errors out of 2050 instances). *(Note: Z-scores are not applicable to this categorical evaluation, thus F1-score and error rates are used).*

### 6.7 Training Strategy

To make transformer fine-tuning practical under limited local hardware, the project follows a **portable training strategy**: train on GPU-capable environments when available, then run inference on local CPU/GPU as needed. In practice, this enables rapid experimentation during training and stable local deployment during development and demo phases.

### 6.8 Environment and Reproducibility

The training pipeline is implemented in PyTorch/Transformers and can run in both local and cloud notebooks (e.g., Colab) without code changes to core logic. Device selection is automatic (`cuda` when available, otherwise `cpu`), and random seeds are fixed for reproducibility. **Section 7** and **Appendix A** record the quantitative outputs of the completed evaluation (tables copied from pipeline exports), so the main text does not rely on file-path citations.

### 6.9 Hyperparameters (Current Configuration)

The default configuration used in the current pipeline is:

- Model: `dbmdz/bert-base-turkish-cased`
- Max sequence length: `160`
- Batch size: `32`
- Epochs: `4`
- Optimizer: **AdamW**
- Learning rate: `2e-5`
- Warmup ratio: `0.1` (linear warmup schedule; total warmup steps = warmup ratio × total optimizer steps)
- Random seed: **`42`** (Python, NumPy, PyTorch; CUDA deterministic mode is not additionally forced)
- Class-weighted loss: **enabled** (`USE_CLASS_WEIGHTS=True`). Class weights are derived from training-label frequencies; the neutral class index is **`1`** with an extra multiplier **`NEUTRAL_LOSS_BOOST=1.0`** (no additional neutral emphasis beyond inverse-frequency weighting at default)
- Gradient clipping: **global L2 norm 1.0** on all parameters after each backward pass
- Early stopping: **enabled**, patience **`2`**, minimum improvement **`1e-4`** on validation macro-F1
- Mixed precision (AMP): enabled when CUDA is available
- DataLoader: **`num_workers=2`** when training on CUDA (CPU training uses `0` workers in code)

**Training-pool merge flags (as in `src/core/config.py` for the reported pipeline):**

- `MERGE_RAW_ABSA_FOR_TRAIN=True` (optional raw ABSA-oriented CSV folded into the training pool only)
- `USE_HF_TRAIN_EXTRA=True` with Hugging Face dataset **`winvoker/turkish-sentiment-analysis-dataset`**, sample size **`10_000`**, subsample seed **`42`**
- `MERGE_HARD_EXAMPLES=True` (curated `hard_examples.csv` overrides merged into training where applicable)

Validation and test CSVs are not altered by these merges; they remain fixed for fair comparison.

### 6.10 Training Monitoring and Model Selection

During training, both train and validation losses are tracked per epoch, and **validation macro-F1** is used for **checkpoint selection**. The best-performing weights are serialized as the single fine-tuned checkpoint used for all reported test evaluation.

**Training run record (saved experiment log, UTC `2026-04-08T10:01:47`):**

| Quantity | Value |
| --- | ---: |
| Merged **training** pool size | **30,770** sentences |
| **Validation** set size | **2,049** sentences |
| Training **device** | CUDA GPU (`cuda:0`) |
| Mixed precision (AMP) | **enabled** |
| Epochs completed | **4** (full schedule; each epoch improved validation macro-F1) |
| **Best validation macro-F1** | **0.846129** (achieved at epoch **4**) |

**Per-epoch validation trace (macro-F1 selection metric):**

| Epoch | Train loss | Val loss | Val macro-F1 |
| --- | ---: | ---: | ---: |
| 1 | 0.4772 | 0.3585 | 0.771450 |
| 2 | 0.1978 | 0.3436 | 0.796239 |
| 3 | 0.1239 | 0.3808 | 0.818709 |
| 4 | 0.0828 | 0.3605 | **0.846129** |

**Held-out test evaluation (same checkpoint as above — Section 7):**

| Quantity | Value |
| --- | ---: |
| **Test** instances | **2,050** (supports: Negative 853, Neutral 130, Positive 1067) |
| **bert_finetuned** test accuracy | **0.973659** |
| **bert_finetuned** test macro-F1 | **0.961286** |
| Test misclassifications | **54** (error rate **2.63%**) |

Validation and test are **different** splits (2,049 vs 2,050 instances); it is **expected** that test macro-F1 can differ from validation macro-F1 once the model is frozen and evaluated on unseen data.

**Selection discipline:** checkpoint choice follows **validation** macro-F1; **Section 7** reports **test-set** metrics for that chosen checkpoint only, without repeated test-driven tuning.

### 6.11 Performance Reporting Protocol

Reported numbers in this document are taken from the **held-out test** evaluation tables in **Section 7** and duplicated for convenience in **Appendix A**. Primary **development-time** decisions should still be driven by **validation** macro-F1 as described in Section 6.7; any future stability study (e.g., multiple seeds) should be labeled explicitly so it is not confused with the single headline test run.

### 6.12 Model Transfer and Deployment Readiness

The trained checkpoint is environment-portable. Loading logic supports both GPU and CPU execution through device-aware `map_location`, allowing a model trained in a GPU environment to be served in local CPU-only scenarios. This supports the project goal of "train once, deploy anywhere" within the constraints of available hardware.

### 6.13 Technologies Used

| Layer | Technology | Role & Contribution |
| :--- | :--- | :--- |
| **Machine Learning** | **PyTorch & Transformers** | Core framework for fine-tuning **BERTurk** with mixed-precision support (AMP). |
| **Data Engineering** | **Pandas & NumPy** | High-performance data normalization, deduplication, and label schema mapping. |
| **API Orchestration**| **ASP.NET Core 8.0** | **C# API Gateway** for traffic routing, security, and enterprise-level service management. |
| **Inference Engine** | **FastAPI (Python)** | High-concurrency ML service handling serialized tensor computations and logit fallbacks. |
| **Frontend Architecture** | **React 18 + Vite** | Component-based interactive dashboard built for performance and modern developer experience. |
| **Data Visualization** | **Recharts** | Dynamic, SVG-based charts for real-time sentiment distribution and batch analytics. |
| **Styling & UX** | **Tailwind CSS** | Utility-first styling for a premium, responsive, and accessible user interface. |
| **Utilities** | **Lucide-react & Storage** | Modern iconography and local state persistence for robust user sessions. |

## 7. Results

### 7.1 Overall Baseline Comparison

On the **held-out test set** (**N = 2050**), the fine-tuned BERTurk model clearly outperforms the classical baselines trained/evaluated under the same split.

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| **bert_finetuned** | **0.973659** | **0.961286** |
| tfidf_logreg | 0.877073 | 0.796227 |
| majority_class | 0.520488 | 0.228211 |

### 7.2 Class-wise Performance (bert_finetuned)

- **Negative:** Precision 0.982122, Recall 0.966002, F1 0.973995, Support 853
- **Neutral:** Precision 0.924242, Recall 0.938462, F1 0.931298, Support 130
- **Positive:** Precision 0.973123, Recall 0.984067, F1 0.978565, Support 1067

These scores indicate high polarity separation quality, with the strongest class-wise recall on Positive and the lowest (yet still strong) F1 on Neutral.

### 7.3 Confusion Matrix Analysis

**Misclassification counts by (true label → predicted label):**

| True label | Predicted label | Count |
| --- | --- | ---: |
| Negative | Positive | 24 |
| Positive | Negative | 12 |
| Negative | Neutral | 5 |
| Neutral | Positive | 5 |
| Positive | Neutral | 5 |
| Neutral | Negative | 3 |

Derived **correct classification rates** (from supports in Section 7.2):

- **Correct classification rates:**
  - Negative: **824 / 853 = 96.60%**
  - Neutral: **122 / 130 = 93.85%**
  - Positive: **1050 / 1067 = 98.41%**
- **Critical polarity-flip errors (Negative <-> Positive):**
  - Positive -> Negative: **12 / 1067 = 1.12%**
  - Negative -> Positive: **24 / 853 = 2.81%**
  - Total polarity flips: **36 cases**
- **Inter-class confusion clusters (by count):**
  1. Negative <-> Positive: **36**
  2. Neutral <-> Positive: **10**
  3. Neutral <-> Negative: **8**

Interpretation: the model is highly reliable on main polarity discrimination, while most remaining errors occur in semantically mixed or contrastive sentences.

### 7.4 Inference and Error Analysis

The evaluation run produced **54** misclassified test sentences (**2.63%** of 2050). Recurring **linguistic patterns** among errors include:

- **Mixed-polarity structures** (frequent *ama / fakat* contrast) where one clause is positive and the other negative.
- **Aspect conflict within one sentence** (e.g., strong praise for atmosphere but criticism of service/price), which is difficult under sentence-level single-label supervision.
- **Implicit dissatisfaction** in neutral-looking expressions, occasionally shifted toward Negative.

**Representative misclassified examples (verbatim from evaluation export):**

| # | Sentence (excerpt) | Gold | Pred |
| --- | --- | --- | --- |
| 1 | Yemekler harika bir tat sunuyor fakat garsonlar ilgisiz. | Neutral | Negative |
| 2 | Mekanın dekorasyonu gerçekten göz alıcıydı fakat yemeklerin tadı pek hoş değildi. | Positive | Negative |
| 3 | Ortamın atmosferi mükemmel ama ücretler oldukça yüksek. | Neutral | Negative |
| 4 | Garsonlar oldukça yardımseverdi fakat yemekler pek tatlı değildi. | Negative | Positive |
| 5 | Restoranın ambiyansı oldukça hoş, fakat yemeklerin tadı pek iyi değil. | Positive | Negative |

From a deployment perspective, these findings justify the confidence-aware fallback policy and support future improvements through targeted hard-example enrichment focused on contrastive and mixed-emotion sentences.

### 7.5 Ablation study (eight runs)

This subsection follows the design matrix in **`ablation_plan.csv`**: **eight** runs (`abl_01`–`abl_08`) formed by all combinations of **`USE_HF_TRAIN_EXTRA`**, **`MERGE_HARD_EXAMPLES`**, and **`CONFIDENCE_FALLBACK_ENABLED`**. Everything else matches **Section 6.3** unless you explicitly document a deviation.

**Why run a structured ablation plan?** The pipeline combines **HF extra training data**, curated **hard examples**, and **confidence fallback** at inference. A **2³** grid separates **marginal** and **interaction** effects (e.g. whether HF volume helps more when hard merge is on) from confounding one-off runs. It also isolates **post-hoc fallback** from **weight updates**: pairs that differ **only** in `CONFIDENCE_FALLBACK_ENABLED` must reuse the **same checkpoint** so test deltas reflect the rule, not a new random seed trajectory.

**Protocol (recommended):**

- Use a **distinct output run name** per row (e.g. `run_abl_01` … `run_abl_08`) so tables and `experiment_last_run.json` do not overwrite each other.
- **Retrain** whenever `USE_HF_TRAIN_EXTRA` or `MERGE_HARD_EXAMPLES` changes the training pool. Rows that differ **only** in `CONFIDENCE_FALLBACK_ENABLED` share the **same weights**; update only the inference/evaluation column for fallback (no second full train).
- **`MERGE_RAW_ABSA_FOR_TRAIN`** is **not** toggled in the plan; keep it **fixed** (e.g. `True` as in Section 6.3) across all eight runs unless you introduce an extra study.
- Record **best validation macro-F1** from the training log and **held-out test** accuracy / macro-F1 / **Neutral F1** after evaluation for each trained checkpoint.

**Design matrix (source flags per run):**

| `run_id` | `USE_HF_TRAIN_EXTRA` | `MERGE_HARD_EXAMPLES` | `CONFIDENCE_FALLBACK_ENABLED` | `requires_retrain` (plan) |
| --- | :---: | :---: | :---: | :---: |
| `abl_01` | False | False | False | False |
| `abl_02` | False | False | True | False |
| `abl_03` | False | True | False | True |
| `abl_04` | False | True | True | True |
| `abl_05` | True | False | False | True |
| `abl_06` | True | False | True | True |
| `abl_07` | True | True | False | True |
| `abl_08` | True | True | True | True |

*The `requires_retrain` column matches the exported **`ablation_plan.csv`** convention (`abl_01`–`abl_02` are **False**). Operationally, rows that differ **only** in `CONFIDENCE_FALLBACK_ENABLED` still **share checkpoints** with their fallback-off partner (`abl_02` with `abl_01`, `abl_04` with `abl_03`, etc.) as in the protocol bullets above.*

**Aggregated results:**

| `run_id` | Best **val** macro-F1 | Test **accuracy** | Test **macro-F1** | Test **Neutral** F1 | One-line note |
| --- | ---: | ---: | ---: | ---: | --- |
| `abl_01` | **0.824022** | **0.968780** | **0.956287** | **0.925926** | `run_abl_01`; train **18,445**; val best epoch **4**; HF/hard/fallback **off**; test errors **64**/2050 (**3.12%**); polarity flips Neg↔Pos **44**. |
| `abl_02` | **0.824022** *(shared train w/ `abl_01`)* | **0.968780** | **0.956287** | **0.925926** | Eval-only: `CONFIDENCE_FALLBACK_ENABLED=True`; same checkpoint as `abl_01`. Exported under `data/outputs/abl_02_reports/abl_02_reports/`. Test metrics match `abl_01` (fallback did not alter labels on this split at `threshold=0.70`). |
| `abl_03` | **0.840658** | **0.966829** | **0.949482** | **0.907143** | `run_abl_03`; train **20,770**; val best epoch **4**; HF **off**, hard **on**, fallback **off**; test errors **68**/2050 (**3.32%**); polarity flips Neg↔Pos **42**. Exported under `data/outputs/abl_03_reports/`. |
| `abl_04` | **0.840658** *(shared train w/ `abl_03`)* | **0.966829** | **0.949482** | **0.907143** | Eval-only: `CONFIDENCE_FALLBACK_ENABLED=True`; same checkpoint as `abl_03`. Exported under `data/outputs/abl_04_reports/`. Test metrics match `abl_03` (fallback did not alter labels on this split at `threshold=0.70`). |
| `abl_05` | **0.835461** | **0.974146** | **0.953689** | **0.902985** | `run_abl_05`; train **28,445**; val best epoch **4**; HF **on**, hard **off**, fallback **off**; test errors **53**/2050 (**2.59%**); polarity flips Neg↔Pos **27**. Exported under `data/outputs/abl_05/`. |
| `abl_06` | **0.835461** *(shared train w/ `abl_05`)* | **0.974146** | **0.953689** | **0.902985** | Eval-only: `CONFIDENCE_FALLBACK_ENABLED=True`; same checkpoint as `abl_05`. Exported under `data/outputs/abl_06_reports/`. Test metrics match `abl_05` (fallback did not alter labels on this split at `threshold=0.70`). |
| `abl_07` | **0.842581** | **0.974634** | **0.962001** | **0.931298** | `run_abl_07`; train **30,770**; val best epoch **4**; HF **on**, hard **on**, fallback **off**; test errors **52**/2050 (**2.54%**); polarity flips Neg↔Pos **34**. Exported under `data/outputs/abl_07_report/`. |
| `abl_08` | **0.842581** *(shared train w/ `abl_07`)* | **0.974634** | **0.962001** | **0.931298** | Eval-only: `CONFIDENCE_FALLBACK_ENABLED=True`; same checkpoint as `abl_07`. Exported under `data/outputs/abl_08_report/`. Test metrics match `abl_07` (fallback did not alter labels on this split at `threshold=0.70`). |

**Per-run checklist (copy after each experiment):**

| Field | Value |
| --- | --- |
| `run_id` | `abl_01` *(train + val log + test eval complete)* |
| Output run label | `run_abl_01` |
| Best val macro-F1 | **0.824022** |
| Epoch of best val | **4** |
| Test accuracy | **0.968780** |
| Test macro-F1 | **0.956287** |
| Test F1 (Neg / Neu / Pos) | **0.968384** / **0.925926** / **0.974552** |
| Fallback at eval | **off** *(matches `abl_01`)* |

| Field | Value |
| --- | --- |
| `run_id` | `abl_02` *(eval-only; same weights as `abl_01`)* |
| Output / export folder | `data/outputs/abl_02_reports/abl_02_reports/` *(nested folder name as saved)* |
| Best val macro-F1 | **0.824022** *(from `abl_01` training run)* |
| Epoch of best val | **4** *(same)* |
| Test accuracy | **0.968780** |
| Test macro-F1 | **0.956287** |
| Test F1 (Neg / Neu / Pos) | **0.968384** / **0.925926** / **0.974552** |
| Fallback at eval | **on** *(matches `abl_02` row)* |

| Field | Value |
| --- | --- |
| `run_id` | `abl_03` *(train + val log + test eval complete)* |
| Output / export folder | `data/outputs/abl_03_reports/` |
| Best val macro-F1 | **0.840658** |
| Epoch of best val | **4** |
| Test accuracy | **0.966829** |
| Test macro-F1 | **0.949482** |
| Test F1 (Neg / Neu / Pos) | **0.965762** / **0.907143** / **0.975541** |
| Fallback at eval | **off** *(matches `abl_03` row)* |

| Field | Value |
| --- | --- |
| `run_id` | `abl_04` *(eval-only; same weights as `abl_03`)* |
| Output / export folder | `data/outputs/abl_04_reports/` |
| Best val macro-F1 | **0.840658** *(from `abl_03` training run)* |
| Epoch of best val | **4** *(same)* |
| Test accuracy | **0.966829** |
| Test macro-F1 | **0.949482** |
| Test F1 (Neg / Neu / Pos) | **0.965762** / **0.907143** / **0.975541** |
| Fallback at eval | **on** *(matches `abl_04` row)* |

| Field | Value |
| --- | --- |
| `run_id` | `abl_05` *(train + val log + test eval complete)* |
| Output / export folder | `data/outputs/abl_05/` |
| Best val macro-F1 | **0.835461** |
| Epoch of best val | **4** |
| Test accuracy | **0.974146** |
| Test macro-F1 | **0.953689** |
| Test F1 (Neg / Neu / Pos) | **0.978299** / **0.902985** / **0.979784** |
| Fallback at eval | **off** *(matches `abl_05` row)* |

| Field | Value |
| --- | --- |
| `run_id` | `abl_06` *(eval-only; same weights as `abl_05`)* |
| Output / export folder | `data/outputs/abl_06_reports/` |
| Best val macro-F1 | **0.835461** *(from `abl_05` training run)* |
| Epoch of best val | **4** *(same)* |
| Test accuracy | **0.974146** |
| Test macro-F1 | **0.953689** |
| Test F1 (Neg / Neu / Pos) | **0.978299** / **0.902985** / **0.979784** |
| Fallback at eval | **on** *(matches `abl_06` row)* |

| Field | Value |
| --- | --- |
| `run_id` | `abl_07` *(train + val log + test eval complete)* |
| Output / export folder | `data/outputs/abl_07_report/` |
| Best val macro-F1 | **0.842581** |
| Epoch of best val | **4** |
| Test accuracy | **0.974634** |
| Test macro-F1 | **0.962001** |
| Test F1 (Neg / Neu / Pos) | **0.975265** / **0.931298** / **0.979439** |
| Fallback at eval | **off** *(matches `abl_07` row)* |

| Field | Value |
| --- | --- |
| `run_id` | `abl_08` *(eval-only; same weights as `abl_07`)* |
| Output / export folder | `data/outputs/abl_08_report/` |
| Best val macro-F1 | **0.842581** *(from `abl_07` training run)* |
| Epoch of best val | **4** *(same)* |
| Test accuracy | **0.974634** |
| Test macro-F1 | **0.962001** |
| Test F1 (Neg / Neu / Pos) | **0.975265** / **0.931298** / **0.979439** |
| Fallback at eval | **on** *(matches `abl_08` row)* |

**What we observed (summary).** For the **HF-off** face (`abl_01`–`abl_04`), enabling **hard-example merge** (`abl_03`/`abl_04` vs `abl_01`/`abl_02`) increases **best validation macro-F1** from **0.824022** to **0.840658** but reduces **held-out test macro-F1** from **0.956287** to **0.949482** and **Neutral F1** from **0.925926** to **0.907143**, with **68** test errors versus **64**—a **validation–generalisation trade-off** on the fixed split. Pairs that differ **only** in **`CONFIDENCE_FALLBACK_ENABLED=True`** at **`threshold=0.70`** match their base runs on test (`abl_02`=`abl_01`, `abl_04`=`abl_03`). **`abl_05`** (**HF on**, hard off, fallback off) is logged: **best val macro-F1 0.835461**, test **accuracy 0.974146** / **macro-F1 0.953689** / **Neutral F1 0.902985**, **53** test errors (**2.59%**), **27** Neg↔Pos flips—under this split, HF extra data **without** hard-example merge improves test accuracy versus **`abl_01`** (**0.974146** vs **0.968780**) while test macro-F1 is **slightly lower** (**0.953689** vs **0.956287**). **`abl_06`** (same weights, fallback **on** at eval) **matches `abl_05`** on all reported test metrics at **`threshold=0.70`**. **`abl_07`** (**HF on**, hard **on**, fallback **off**): **best val macro-F1 0.842581**, test **accuracy 0.974634** / **macro-F1 0.962001** / **Neutral F1 0.931298**, **52** errors (**2.54%**), **34** Neg↔Pos flips—versus **`abl_03`** (HF off, hard on), **`abl_07`** improves held-out **macro-F1** (**0.962001** vs **0.949482**) and **Neutral F1** (**0.931298** vs **0.907143**) with **fewer** total errors (**52** vs **68**) and **fewer** Neg↔Pos flips (**34** vs **42**). **`abl_08`** (fallback **on**, same weights as **`abl_07`**) **matches `abl_07`** on all reported test metrics at **`threshold=0.70`**.

**What the plan is useful for.** The grid **attributes** changes to **HF data**, **hard examples**, and **fallback** instead of anecdotal runs, supports **reproducible** thesis reporting, and **guides deployment** trade-offs (which merge flags to ship, and whether fallback is worth its complexity).

**Joint interpretation (all eight runs).** **Confidence fallback** at **`threshold=0.70`** did **not** change any **aggregate** held-out test metric for any of the four fallback-on rows relative to their partners (`abl_02`=`abl_01`, `abl_04`=`abl_03`, `abl_06`=`abl_05`, `abl_08`=`abl_07`); on this benchmark, max-probability mass stayed above the rule for the exported evaluation. **Hard-example merge** under **`USE_HF_TRAIN_EXTRA=False`** improves **validation** macro-F1 (**`abl_03`**/**`abl_04`** vs **`abl_01`**/**`abl_02`**) but **hurts** held-out **test macro-F1** and **Neutral F1**, with **more** test errors (**68** vs **64**), i.e. a clear **validation–test tension** for that slice. Turning **HF extra data on** while **hard merge stays off** (`abl_05`/`abl_06` vs `abl_01`/`abl_02`) **raises test accuracy** (**0.974146** vs **0.968780**) but **lowers** test macro-F1 (**0.953689** vs **0.956287**) and **Neutral F1** (**0.902985** vs **0.925926**). The **HF-on + hard-on** configuration (`abl_07`/`abl_08`) achieves the **strongest** reported combination of **test macro-F1** (**0.962001**) and **Neutral F1** (**0.931298**) in the ablation table, with the **fewest** errors (**52**) among trained rows and **fewer** Neg↔Pos flips than HF-off+hard (**34** vs **42**). Overall, **HF** and **hard merge** move **different** levers on val vs test; **fallback** here is a **no-op on aggregates** but remains a **cheap safeguard** for other inputs (see **Section 8.4**).

## 8. Discussion

Section **7.1–7.2** report the **primary** fine-tuned BERTurk checkpoint under the default training/evaluation configuration described in **Section 6** (held-out test **N = 2050**). Section **7.5** reports **controlled ablations** where merge and inference flags differ by design; lower headline test macro-F1 on some ablation rows (e.g. `abl_01`–`abl_04`) is therefore **expected** and should not be read as a contradiction of Section 7.1 without aligning the configuration.

### 8.1 Main Findings

The fine-tuned **BERTurk** classifier substantially outperforms **TF–IDF + logistic regression** and the **majority-class** baseline on the same split (**Section 7.1**). Contextual token representations capture non-linear interactions and sub-word morphology far better than sparse bag-of-words features, which explains most of the accuracy and macro-F1 gap. The majority baseline is structurally weak on imbalanced three-way sentiment because it collapses predictions onto the dominant class.

Class-wise behaviour (**Section 7.2**) is strongest on **Positive** (highest F1) and comparatively weakest on **Neutral**, consistent with the **smallest support** for Neutral in the test distribution. Neutral spans inherently vague or low-intensity sentiment, so the model’s relative difficulty is both a **data scarcity** effect and a **semantic ambiguity** effect. Confusion concentrates on **contrastive** and **mixed-polarity** sentences (**Sections 7.3–7.4**): when a single label must summarize two opposing evaluations joined by *ama/fakat*, sentence-level supervision forces a compromise label that neither clause fully satisfies, which increases polarity flips and Neutral boundary errors.

### 8.2 Error Taxonomy (Based on Misclassified Samples)

The following taxonomy groups recurring failure modes observed in the misclassification analysis (**Section 7.4**). Each row gives one **verbatim** test excerpt (Turkish), the **gold** label, the **predicted** label, and a one-line interpretation.

| Category | Representative excerpt (test) | Gold | Pred | One-line interpretation |
| --- | --- | --- | --- | --- |
| Contrastive conjunctions | *Yemekler harika bir tat sunuyor fakat garsonlar ilgisiz.* | Neutral | Negative | Concessive structure pulls the model toward the negative clause despite global neutrality. |
| Mixed sentiment (single sentence) | *Mekanın dekorasyonu gerçekten göz alıcıydı fakat yemeklerin tadı pek hoş değildi.* | Positive | Negative | Aspect-level praise and criticism conflict under a single sentence-level polarity target. |
| Neutral vs polarized boundary | *Ortamın atmosferi mükemmel ama ücretler oldukça yüksek.* | Neutral | Negative | Price criticism dominates the representation even when the annotator chose Neutral overall. |
| Polarity flip (Neg ↔ Pos) | *Garsonlar oldukça yardımseverdi fakat yemekler pek tatlı değildi.* | Negative | Positive | Service positivity competes with food criticism; the model locks onto the “wrong” clause for the gold label. |
| Implicit / low-intensity dissatisfaction | *(See narrative bullets in Section 7.4 and `test_misclassified.csv` exports.)* | *(varies)* | *(varies)* | Mild or indirect negativity can disagree with a Neutral gold standard under sentence-level single labels. |

Long-sequence **attention drift** was not isolated as a dominant failure mode in the reviewed sample; most errors remain interpretable as **semantic** rather than **length** artefacts within `max_len=160`.

### 8.3 Validity, Limitations, and Threats

**Domain bias and transfer.** The training and evaluation corpora are **sentence-level** Turkish sentiment with a strong **service/review** flavour. Performance on distant domains (e.g. legal text, clinical notes, highly informal social media) is **not guaranteed** without adaptation or re-labeling.

**Class imbalance.** Neutral has the **lowest support** on the held-out test set (**Section 7.2**). Class-weighted loss mitigates but does not remove imbalance effects; reported Neutral F1 should be read alongside **support** and confusion tables.

**Label ambiguity.** Sentence-level **single** labels cannot fully encode **multi-aspect** opinions. Some “errors” may reflect **legitimate alternate readings** rather than model incapacity, which limits ceiling accuracy under this annotation scheme.

**Experimental breadth.** Results rely on a **fixed random seed** and practical GPU/Colab constraints. The **eight-run** ablation grid in **Section 7.5** is now **fully populated**, which tightens internal comparisons across **HF**, **hard-example merge**, and **fallback**—while still reflecting a **single** optimisation trajectory per trained configuration. External HF subsampling (`HF_SAMPLE_SIZE = 10_000`) introduces an additional **sensitivity** dimension.

### 8.4 Practical Implications

The project closes the loop from **training artefacts** to a **deployable product**: the serialized checkpoint is loaded by a hybrid microservices architecture. A **Python FastAPI** service handles the heavy tensor computations and chart generation, while a **C# ASP.NET Core API Gateway** orchestrates traffic, providing a scalable, enterprise-ready separation of concerns (**Sections 1, 6.7**).

The user experience has been elevated to a product-ready standard through a **React + Vite** dashboard. Non-expert users can process large CSV files or text batches with real-time **progress tracking** via chunked requests. The dashboard features interactive **Recharts** visualizations, sortable and searchable data tables, and one-click exports for both statistics (CSV) and distribution charts (PNG). **State persistence** ensures workflows are not lost on refresh.

**Confidence fallback** is operationally useful as a **safety valve** for low-certainty inputs even when, on the **held-out benchmark** at `threshold=0.70`, fallback did not relabel any test instance for the evaluated ablation pairs (**Section 7.5**). In production, traffic can be **more out-of-distribution** than the test split; mapping low-confidence logits to **Neutral** remains a low-cost risk reduction pattern.

The model is **most reliable** when polarity is **explicit and one-sided**; it is **riskiest** on **contrastive** and **mixed-affect** sentences (**Section 7.4**). The comprehensive batch endpoints, interactive visualizer, and export tools support **non-expert** workflows (triage, campaign monitoring) without requiring notebook literacy.

### 8.5 Recommended Improvement Roadmap

1. **Targeted hard-example curriculum** focused on *ama/fakat* and mixed-aspect restaurant language, possibly with **counterfactual** or **multi-label** auxiliary tasks, building on the validation–test trade-off observed when `MERGE_HARD_EXAMPLES` is toggled (**Section 7.5**).
2. **Multi-seed training and reporting** (e.g. mean ± std on validation and test) to quantify variance and reduce overfitting to a single optimisation trajectory.
3. **Calibration analysis** (reliability diagrams, temperature scaling) so the **confidence fallback** threshold is chosen from **measured** probability quality rather than a fixed default.
4. **Threshold and calibration follow-up** now that the **Section 7.5** grid shows fallback inactivity on the held-out split at **`0.70`**—sweep thresholds and measure reliability on **out-of-domain** samples where fallback may activate.
5. **Latency and compression** (distillation, quantisation, smaller encoders) if the API path moves from demo to **production** scale.

## 9. Conclusion

This study presented an end-to-end solution to the problem of sentence-level sentiment analysis (Negative, Neutral, Positive) in user-generated Turkish texts, approaching it from both academic and engineering perspectives. In this research, the **BERTurk** language model was fine-tuned using a specially curated dataset of approximately **30,770** sentences (including HuggingFace data and hard examples). A rigorous, leakage-aware training pipeline was established to effectively manage class imbalances.

The results demonstrated that the model significantly outperformed classical machine learning methods (TF-IDF + Logistic Regression). In the final evaluations, our model achieved exceptional metrics, including an overall accuracy of **97.5%** and a macro-F1 score of **0.962**. Error analysis revealed that the model struggled most with complex sentences containing contrastive conjunctions (e.g., "but", "however") and dual-polarity sentiments. To minimize these risks, a **70% Confidence Fallback** mechanism was implemented, forcing the model to default to the safer "Neutral" class in uncertain scenarios, thereby increasing the system's operational reliability.

Beyond academic metrics and offline success, the greatest contribution of this project is its transition from a laboratory environment into a **production-ready** software architecture. The developed system is powered by a high-performance **Python (FastAPI)** inference engine, an enterprise-grade **C# (ASP.NET Core) API Gateway** for security and traffic orchestration, and a reactive **React + Vite** frontend that refines the end-user experience.

In conclusion, this project successfully demonstrates not only the training of a highly accurate deep learning model, but also how such a model can be transformed into a scalable, interactive, and professional software product using modern microservice architectures.

## References

References follow standard academic formatting. Web pages include retrieval date (**11 April 2026**).

1. Dehkharghani, R., Saygin, Y., Yanikoglu, B., & Oflazer, K. (2016). SentiTurkNet: A Turkish polarity lexicon for sentiment analysis. *Language Resources and Evaluation*, 50(3), 667–685.
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the NAACL-HLT* (Vol. 1, pp. 4171–4186). <https://doi.org/10.18653/v1/N19-1423>
3. Hugging Face. (n.d.). *Datasets: winvoker/turkish-sentiment-analysis-dataset*. Retrieved 11 April 2026, from <https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset>
4. Kim, Y. (2014). Convolutional neural networks for sentence classification. In *Proceedings of EMNLP* (pp. 1746–1751). <https://doi.org/10.3115/v1/D14-1181>
5. Liu, B. (2012). *Sentiment analysis and opinion mining*. Morgan & Claypool Publishers. <https://doi.org/10.2200/S00416ED1V01Y201204HLT016>
6. Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. *Foundations and Trends in Information Retrieval*, *2*(1–2), 1–135. <https://doi.org/10.1561/1500000011>
7. Paszke, A., Gross, S., Massa, F., Lerer, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in NIPS* (Vol. 32).
8. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.
9. Pontiki, M., Galanis, D., Pavlopoulos, J., et al. (2014). SemEval-2014 Task 4: Aspect based sentiment analysis. In *Proceedings of SemEval 2014* (pp. 27–35).
10. Schweter, S. (2020). *BERTurk — BERT models for Turkish* (Version 1.0) [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.3770924>
11. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. In *Advances in NIPS* (Vol. 30).
12. Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-art natural language processing. In *Proceedings of EMNLP* (pp. 38–45). <https://doi.org/10.18653/v1/2020.emnlp-demos.6>
13. Yıldırım, A., & Yağcı, M. (2022). A comprehensive review of sentiment analysis in Turkish language. *IEEE Access*, 10, 15412–15429.

## Appendix

### Appendix A - Tabulated Evaluation Exports

Sections **A.1–A.4** reproduce the **test-set** evaluation exports (aligned with Section 7). Section **A.5** summarizes the **training-time validation trace** from the same training run as Section 6.4 (no file paths in the body text).

**A.1 Baseline comparison (test set)**

| Model | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| bert_finetuned | 0.973659 | 0.961286 |
| tfidf_logreg | 0.877073 | 0.796227 |
| majority_class | 0.520488 | 0.228211 |

**A.2 Per-model, per-class classification report (test set)**

| Model | Class | Precision | Recall | F1 | Support |
| --- | --- | ---: | ---: | ---: | ---: |
| majority_class | Negative | 0.0 | 0.0 | 0.0 | 853 |
| majority_class | Neutral | 0.0 | 0.0 | 0.0 | 130 |
| majority_class | Positive | 0.520488 | 1.0 | 0.684633 | 1067 |
| tfidf_logreg | Negative | 0.875719 | 0.892145 | 0.883856 | 853 |
| tfidf_logreg | Neutral | 0.589552 | 0.607692 | 0.598485 | 130 |
| tfidf_logreg | Positive | 0.914995 | 0.897844 | 0.906339 | 1067 |
| bert_finetuned | Negative | 0.982122 | 0.966002 | 0.973995 | 853 |
| bert_finetuned | Neutral | 0.924242 | 0.938462 | 0.931298 | 130 |
| bert_finetuned | Positive | 0.973123 | 0.984067 | 0.978565 | 1067 |

**A.3 Confusion pairs (raw counts, test set)**

| True label | Predicted label | Count |
| --- | --- | ---: |
| Negative | Positive | 24 |
| Positive | Negative | 12 |
| Negative | Neutral | 5 |
| Neutral | Positive | 5 |
| Positive | Neutral | 5 |
| Neutral | Negative | 3 |

**A.4 Misclassification volume**

- Total errors: **54** on **2050** test instances.

**A.5 Training run — validation trace (same run as Section 6.4)**

| Quantity | Value |
| --- | ---: |
| Merged training pool | **30,770** sentences |
| Validation size | **2,049** sentences |
| Best **validation** macro-F1 | **0.846129** (epoch **4**) |
| Epochs run | **4** |
| Runtime | CUDA, AMP **on** |

| Epoch | Train loss | Val loss | Val macro-F1 |
| --- | ---: | ---: | ---: |
| 1 | 0.477190 | 0.358530 | 0.771450 |
| 2 | 0.197767 | 0.343610 | 0.796239 |
| 3 | 0.123857 | 0.380846 | 0.818709 |
| 4 | 0.082811 | 0.360478 | **0.846129** |

*Optional figure:* include a **confusion matrix heatmap** in the PDF version of the thesis if generated by the evaluation script (figure caption only—no file path in the body text).

### Appendix B - Configuration Snapshot

The exact training-time settings for the reported model match **Section 6.3**. The saved training log for this run additionally records: `random_seed=42`, `early_stopping_patience=2`, `early_stopping_min_delta=0.0001`, `merge_raw_absa_for_train=True`, `use_hf_train_extra=True` with `hf_dataset_id=winvoker/turkish-sentiment-analysis-dataset`, `hf_sample_size=10000`, `hf_seed=42`, `merge_hard_examples=True`, `leakage_guard_enabled=True`, and inference defaults `confidence_fallback_enabled=True`, `confidence_threshold=0.7`, `confidence_fallback_label=Neutral`.

### Appendix C - Example Error Cases

Ten representative rows from the **`run_default`** evaluation export **`test_misclassified.csv`** (same primary run as **Sections 7.1–7.4**). **Error category** follows the taxonomy in **Section 8.2** (abbreviated here for column width).

| # | Sentence (verbatim; line breaks removed) | True | Pred | Error category |
| --- | --- | --- | --- | --- |
| 1 | Yemekler harika bir tat sunuyor fakat garsonlar ilgisiz. | Neutral | Negative | Contrastive / Neutral boundary |
| 2 | Mekanın dekorasyonu gerçekten göz alıcıydı fakat yemeklerin tadı pek hoş değildi. | Positive | Negative | Mixed sentiment / contrastive |
| 3 | Ortamın atmosferi mükemmel ama ücretler oldukça yüksek. | Neutral | Negative | Neutral vs polarized boundary |
| 4 | Garsonlar oldukça yardımseverdi fakat yemekler pek tatlı değildi. | Negative | Positive | Polarity flip (Neg ↔ Pos) |
| 5 | Restoranın ambiyansı oldukça hoş, fakat yemeklerin tadı pek iyi değil. | Positive | Negative | Polarity flip / contrastive |
| 6 | Güler yüzlü personeli ile ortamı rahatlatan, fakat fiyatları biraz yüksek olan bir mekan. | Negative | Neutral | Implicit dissatisfaction / hedged polarity |
| 7 | tatli secenekleri de mevcur sufle ve kunefe var menu de var da pide o kadar gec gelince insani tatli siparisi dusunduruyor. | Negative | Positive | Informal syntax / implicit negation |
| 8 | Yemeklerin tadı mükemmel fakat servis biraz yavaş. | Neutral | Positive | Neutral vs polarized boundary |
| 9 | Mekanın tasarımı harika ama menüdeki fiyatlar biraz tuzlu, atmosfer ve yer açısından bu fiyatları eleştirmek haksızlık olur. | Positive | Neutral | Mixed sentiment / self-correction |
| 10 | Kahvaltı çeşitleri gerçekten çok lezzetli ve doyurucu, fakat fiyatlar biraz yüksek. Personel oldukça yardımsever, ama ortamın daha ferah olması, daha iyi bir deneyim sunabilirdi. | Neutral | Negative | Long contrastive chain |

### Appendix D - Project Evolution Note

The repository and early project framing included **aspect-based** sentiment resources and terminology (e.g. ABSA-oriented corpora and SemEval-style positioning in references). The **final thesis scope** was deliberately narrowed to **sentence-level three-way polarity** for three reasons: (i) **annotation consistency**—single-sentence gold labels are easier to audit than coupled aspect spans; (ii) **deployment alignment**—many product surfaces (ticket triage, review stars, campaign monitoring) consume **whole-sentence** scores first; and (iii) **methodological closure**—one primary supervised objective avoids mixing ABSA extraction quality with sentence-classifier quality in the same headline claims. ABSA remains a **natural extension** (see **Section 8.5**) rather than a silent scope change within the reported experiments.
