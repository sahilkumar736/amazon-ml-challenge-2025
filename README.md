
# Amazon ML Challenge 2025 

---

## 1. Executive Summary

Our team developed a **hybrid deep learning approach** combining **DeBERTa-large-v3** with **engineered features** through a **cross-attention fusion mechanism**.
This architecture leverages **pretrained language understanding** enriched with **domain-specific product attributes**, resulting in robust and generalizable **price prediction performance**.

---

## 2. Methodology Overview

### 2.1 Problem Analysis & Key Observations

Through extensive exploratory data analysis (EDA), we observed that **product pricing** is influenced by both **semantic content** and **explicit attributes**.
Key findings include:

* Price correlations with **quality indicators** (organic, gourmet, gluten-free)
* Importance of **text structure** (length, punctuation, capitalization ratio)
* Relevance of **packaging characteristics** (bulk size, unit type)

### 2.2 Solution Strategy

**Approach Type:** Hybrid (Pretrained LM + Feature Engineering + Cross-Attention)
**Core Innovation:**
A **two-stream architecture** that fuses DeBERTa’s `[CLS]` embedding with **engineered feature embeddings** via **cross-attention**, capturing complex relationships between product descriptions and structured metadata.

**Dataset Insight:**
As the `log(price)` distribution approximated a **Gaussian**, training was performed on **log(price + 1)**.
Final predictions were obtained via `exp(pred) - 1`.

We also evaluated **CLIP-based image embeddings**, but analysis via **UMAP clustering** revealed poor consistency of embedding clusters with price values. Hence, we relied solely on **text-based modeling**.

**Loss Function:**
We used **Smooth L1 loss** instead of MSE, as it yielded lower SMAPE and more stable training.

---

## 3. Model Architecture

### 3.1 Architecture Overview

| Stage                 | Model/Component                  | SMAPE (Validation) 
| --------------------- | -------------------------------- | ------------------ | -------------- |
| First Approach        | Bert Base                |  49.1      
| Pretraining           | DeBERTa (Regression Task)| 44.2 
| Final Hybrid Training | DeBERTa + Cross-Attention Fusion| 40.1

---

### 3.2 Model Components

#### 🧩 Text Processing Pipeline

* Tokenization: DeBERTa tokenizer
* Preprocessing: text normalization, truncation, special char handling
* Model: DeBERTa-large-v3 (1024 hidden dim)
* Pretraining task: price regression

#### ⚙️ Feature Engineering Pipeline

* **Binary:** `is_organic`, `is_gourmet`, `is_gluten_free`, `is_bulk`, `special_chars`
* **Categorical:** `unit_type`
* **Numeric:** `value`, `num_words`, `num_sentences`, `uppercase_ratio`

**Fusion Strategy:**
Concatenated feature embeddings + DeBERTa `[CLS]` → Cross-Attention → Linear Regression Head

---

### ❌ Other (Failed) Approaches

* **XGBoost / CatBoost** on sparse feature datasets — SMAPE ≈ 56
* **Reinforcement Learning Finetuning:** used SMAPE as a reward on frozen DeBERTa backbone; limited success due to differentiability constraints.

---

## 4. Model Performance

| Model                       | Validation SMAPE | Notes       |
| --------------------------- | ---------------- | ----------- |
| Pretrained DeBERTa Baseline | 43.4             | Text-only   |
| Final Hybrid Model          | 40.1             | 5% hold-out |


---

## 5. Conclusion

Our hybrid model successfully demonstrates that **structured features** can enhance the predictive power of **pretrained language models** for complex **pricing regression tasks**.
The **cross-attention fusion** effectively learns the interplay between **semantic understanding** and **explicit product attributes**, achieving scalable and interpretable performance gains for real-world e-commerce applications.

---

## 📁 Repository Structure

```
Amazon-ML-Challenge-2025-3rd/
│
├── Data/
│   ├── preprocessed_train.csv
│   ├── preprocessed_test.csv
│
├── Preprocess.py         # Data preprocessing
├── Pretraining.py        # DeBERTa pretraining on regression
├── Main_Training.py      # Final hybrid model training
├── Inference.py          # Inference and submission CSV generation
├── README.md
```

---
