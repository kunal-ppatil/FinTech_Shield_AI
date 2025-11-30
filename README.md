# FinTech_Shield_AI
A hybrid AI system for detecting financial fraud. Combines a Deep Learning Autoencoder (to spot anomalies) with XGBoost (to classify threats). Designed for imbalanced data, it features SHAP for explainability and a simulated production API to block/allow transactions in real-time.
# 🛡️ Hybrid Financial Fraud Detection System: Autoencoder & XGBoost

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Autoencoder-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-red)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)

A production-grade architecture for detecting credit card fraud. Unlike standard classifiers that fail on highly imbalanced data (where fraud is <1%), this project uses a **Dual-Stage Hybrid Pipeline**. It combines Unsupervised Deep Learning (to detect anomalies) with Supervised Gradient Boosting (to classify specific fraud patterns).

### We built the code, and GPT helped polish it for clarity.
---

## 🏗️ The Architecture

Traditional models often fail in finance because fraud patterns evolve ("Zero-day attacks"). To solve this, this system utilizes two distinct "brains":

1.  **The Detective (Unsupervised Autoencoder):** Learns the pattern of *normality*. If a transaction is too unique, it flags it as an anomaly.
2.  **The Judge (Supervised XGBoost):** Takes the transaction details *plus* the anomaly score to make a final Block/Allow decision.



---

## 🚀 Step-by-Step Pipeline Explanation

This repository implements the following end-to-end workflow:

### Stage 1: Synthetic Data Generation
**The Challenge:** Real financial data is PII (Personally Identifiable Information) protected.
**The Solution:** We generate a synthetic dataset mimicking real banking transactions using `make_classification`.
* **Imbalance:** 99% Legitimate, 1% Fraud.
* **Complexity:** 30 features (V1...V30) representing PCA-transformed sensitive data.
* **Separation:** We use `class_sep=1.2` to create realistic but distinct fraud clusters.

### Stage 2: The Autoencoder (Anomaly Detection)
**The Logic:** We train a Deep Neural Network to compress and reconstruct *only* legitimate transactions.
* **Training:** The model never sees fraud during training. It only memorizes "Normal" spending.
* **Inference:** When a fraud transaction is fed in, the model fails to reconstruct it accurately.
* **Metric:** We calculate the **Mean Squared Error (MSE)** between input and output. This MSE becomes our **Anomaly Score**.

### Stage 3: Hybrid Feature Engineering
**The Handover:** We do not throw away the Autoencoder's work. Instead, we augment the dataset.
* **Input:** Original 30 Features.
* **New Feature:** `Anomaly_Score` (The reconstruction error from Stage 2).
* **Result:** The dataset passed to the next stage has 31 features. This gives the supervised model a "hint" about how weird the transaction looks.

### Stage 4: Supervised Classification (XGBoost)
**The Logic:** XGBoost is the industry standard for tabular data.
* **Input:** The hybrid dataset (Features + Anomaly Score).
* **Handling Imbalance:** We use `scale_pos_weight` to tell the model to pay 100x more attention to the minority class (Fraud).
* **Outcome:** A probability score (0 to 1) indicating the likelihood of fraud.

### Stage 5: Explainability (SHAP)
**The Compliance:** You cannot block a user without a reason.
* We use **SHAP (SHapley Additive exPlanations)** to break down the "Black Box" decision.
* It generates a beeswarm plot showing exactly which features (e.g., `V14`, `Anomaly_Score`) drove the decision to block a specific card.

### Stage 6: Production Simulation
**The API:** The project concludes with a `process_transaction()` function.
* It simulates a live server endpoint.
* It takes raw transaction data $\rightarrow$ Scales it $\rightarrow$ Gets Anomaly Score $\rightarrow$ Predicts with XGBoost $\rightarrow$ Returns a JSON `BLOCKED` or `APPROVED` response.

---

## 🛠️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/hybrid-fraud-detection.git](https://github.com/yourusername/hybrid-fraud-detection.git)
    cd hybrid-fraud-detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow xgboost shap scikit-learn matplotlib seaborn pandas
    ```

3.  **Run the pipeline:**
    ```bash
    python main.py
    ```

---

## 📊 Results

* **Metric of Choice:** AUPRC (Area Under Precision-Recall Curve). This is preferred over "Accuracy" for imbalanced datasets because 99% accuracy is easy (just predict everything is normal).
* **Performance:** The system achieves high Recall (catching fraud) while maintaining precision (avoiding false positives).
* **Key Insight:** SHAP analysis confirms that the **Autoencoder's Anomaly Score** is consistently one of the top 3 most important features used by XGBoost to make decisions.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
