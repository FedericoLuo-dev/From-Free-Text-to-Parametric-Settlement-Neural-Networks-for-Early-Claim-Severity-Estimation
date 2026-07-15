# From Free Text to Parametric Settlement: Neural Networks for Early Claim Severity Estimation

## 📌 Overview
This project explores the application of Natural Language Processing (NLP) and Deep Learning to the insurance sector. It aims to predict the financial severity (cost) of an insurance claim relying **solely on free-text descriptions** of the damage. The ultimate goal is to enable a **Parametric Triage System** that automatically approves low-complexity claims while routing severe cases to human appraisers.

### 🔄 Data Generation Pipeline
<img width="767" height="165" alt="Data generation pipeline" src="https://github.com/user-attachments/assets/b99e2044-8b74-40cb-a9d7-cd6ac5859663" />


## 🏗️ Architecture & Pipeline
The project follows a complete end-to-end Machine Learning pipeline:

1. **Semantic Extraction:** Utilizing advanced LLM embeddings (Gemini API) to map raw claim descriptions into a high-dimensional vector space (3072 dimensions).
2. **Dimensionality Reduction:** Applying Principal Component Analysis (PCA) to retain 95-98% of the explained variance, optimizing computational efficiency while preserving crucial contextual nuances.
3. **Actuarial Neural Network:** A custom Deep Neural Network built in `PyTorch` designed for continuous regression on highly skewed cost distributions (typical of insurance data).

### 🧠 Deep Neural Network Architecture
<img width="955" height="240" alt="Deep Neural Network Pipeline" src="https://github.com/user-attachments/assets/98a82dcc-0c8b-401a-b14d-0bb825a531f0" />


4. **Parametric Triage:** A business-logic layer that evaluates the neural network's predictions against financial thresholds to trigger automated settlements.

## 🔬 Key Insights & Research Findings
During the development of this model, two critical phenomena were identified and addressed:
* **The "Lexical Inflation" Phenomenon:** Synthetic data generation can lead to overly dramatic vocabulary for minor damages (e.g., describing hail as "blunt force trauma"). The model successfully highlights the sensitivity of embeddings to these lexical choices.
* **The Information Bottleneck:** Forcing a low-dimensional output at the API level destroys subtle contextual mitigations, causing the neural network to degrade from a continuous regressor into a rigid classifier. Maintaining high native dimensionality with dynamic PCA proved essential for accurate pricing.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch
* **Data Processing & ML:** Pandas, NumPy, Scikit-learn (PCA), Text-Embedding
* **NLP / Embeddings:** Google GenAI API 
* **Visualization:** Matplotlib, Seaborn

2. Install dependencies: `pip install -r requirements.txt`
3. Add your API key in the configuration file.
4. Run the main pipeline: `python main.py`
