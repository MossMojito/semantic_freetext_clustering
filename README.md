# 🧭 Business-Steered NLP Clustering: Bridging AI and Business Rules

An enterprise-grade, semi-supervised NLP architecture designed to categorize and route messy, unstructured customer free-text (with full Thai language support) when no labeled training data is available.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991)
![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-UMAP_%7C_HDBSCAN-F7931E)

## 🚨 The Real-World Problem

When dealing with thousands of daily customer inquiries, businesses need to route text to the correct departments (e.g., Billing, Delivery, IT Support). However, traditional machine learning approaches fail in the real world due to two extremes:

1. **The Supervised Trap (The "Cold Start"):** We don't have thousands of manually labeled historical tickets to train a classification model.
2. **The Unsupervised Trap (The "Math vs. Business" Clash):** If we just use pure unsupervised clustering (like K-Means or basic HDBSCAN), the AI groups text based on mathematical similarity, not business logic. It might cluster all "angry sounding" complaints together, completely ignoring that one is a Delivery issue and another is a Payment bug.

## 💡 The Solution: Business-Steered Clustering

This architecture solves the dilemma by decoupling the NLP pipeline into two distinct layers. It uses unsupervised algorithms to find the natural shape of the data, but applies "Centroid Steering" via a YAML configuration file to forcefully pull those clusters into predefined Business Departments.

### The Architecture
* **Layer 1: Broad Intent Classification (LLM)**
  * Uses a zero-shot LLM (GPT-4o-mini) to instantly classify the broad intent (e.g., `complain`, `information`, `service issue`).
* **Layer 2: Granular Topic Clustering (Local NLP Math)**
  * Converts multilingual text (Thai/English) into high-dimensional vectors using the `multilingual-e5-base` sentence transformer.
  * Compresses dimensions using **UMAP** for performance.
  * Clusters dense neighborhoods using **HDBSCAN**.
  * **The Secret Sauce:** Uses Cosine Distance to map the unsupervised clusters directly to explicit business rules defined in `config/domain_config.yaml`.
* **Layer 3: Smart Fallback**
  * If the local math identifies a sentence as "Noise/Outlier", it gracefully falls back to the LLM's predicted intent to ensure 100% routing coverage.

## 📂 Enterprise Project Structure

```text
sentence_similarity-semantic/
├── config/
│   └── domain_config.yaml       # The Business Rulebook (Departments & Topics)
├── data/
│   └── mock_customer_data.csv   # Unstructured customer text inputs
├── models/                      # Serialized .joblib models for fast production inference
├── notebooks/                   # Jupyter notebooks for 2D UMAP visualization
├── src/                         # Core pipeline engine
│   ├── classifier.py            # Layer 1: LLM API integration
│   ├── clusterer.py             # Layer 2: Embeddings, UMAP, HDBSCAN, and Steering
│   ├── fallback.py              # Layer 3: Pandas merge and fallback logic
│   └── utils/
│       └── config_loader.py     # Dynamic YAML parsing
├── fit_pipeline.py              # The Calibrator: Learns patterns and saves models
├── inference_api.py             # The Production Server: Real-time interactive routing
└── requirements.txt             # Lightweight dependencies
```

## 🚀 Quick Start (Local Real-Time Engine)

This project is built with pure Pandas and NumPy for maximum portability. No heavy Spark or Hadoop clusters required.

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Securely Inject API Key**
(Do not use `.env` files to prevent accidental commits)
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

**3. Calibrate the Models (Fit Mode)**
Run this once to learn the data shape and serialize the models to the `models/` directory.
```bash
python fit_pipeline.py
```

**4. Start the Interactive Inference Engine**
Launch the real-time prompt to test the live routing architecture!
```bash
python inference_api.py
```
*Example input: "แอปค้างตอนจ่ายเงิน ตัดบัตรไปแล้วแต่ไม่มีออเดอร์"*

## 📊 Data Visualization
To see the mathematical distances between business clusters, open the `notebooks/01_visualize_clusters.ipynb` file to render a 2D UMAP scatter plot of the AI's semantic groupings.