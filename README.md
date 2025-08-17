
# 🛍️ ShopMind: E-Commerce Recommendation Engine Using Behavioral Pattern Mining

**ShopMind** is a data-driven recommendation system that identifies high-intent user behaviors on e-commerce websites and generates actionable next-best-action suggestions. Built using association rule mining (Apriori algorithm), this project analyzes session-level browsing patterns—such as page visits and time spent—to predict purchase likelihood and recommend targeted interventions.

The solution leverages the [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset) from the UCI Machine Learning Repository and transforms user navigation paths into market-basket-style transactions for frequent pattern analysis.

Deployed via **Streamlit**, the system provides an interactive interface for real-time recommendations, making it suitable for marketing, UX optimization, and conversion rate improvement strategies.

---

## 🎯 Project Objectives

- Discover frequent behavioral patterns that lead to purchases.
- Extract association rules to identify high-conversion user journeys.
- Build a recommendation engine that suggests real-time actions (e.g., show discount, live chat).
- Deliver an accessible web interface for non-technical stakeholders.

---

## 🧩 Key Features

- **Behavioral Binning**: Converts numerical page views and durations into categorical user actions.
- **Apriori Algorithm**: Mines frequent itemsets and generates association rules (e.g., "If Product_High and Duration_Long → Purchase").
- **Actionable Insights**: Translates rules into business recommendations.
- **Interactive Dashboard**: Streamlit app allows users to input session data and receive instant recommendations.
- **Reproducible Environment**: Full Conda setup ensures consistent development and deployment.

---

## 🔍 Methodology

1. **Data Preprocessing**  
   Session features (`Administrative`, `Informational`, `ProductRelated`, durations) are discretized into categorical bins (e.g., Low, Medium, High).

2. **Transaction Encoding**  
   Each session is treated as a "basket" of behavioral items (e.g., `Product_High`, `ProdDur_Long`).

3. **Frequent Pattern Mining**  
   The Apriori algorithm identifies itemsets that frequently co-occur in sessions resulting in purchases.

4. **Rule Generation**  
   Association rules are extracted using metrics like **support**, **confidence**, and **lift** to prioritize high-impact patterns.

5. **Recommendation Logic**  
   Rules with high lift (>1.5) and confidence (>0.3) are used to trigger personalized recommendations.

6. **Web Interface**  
   A Streamlit app enables real-time inference by accepting user inputs and matching them against learned rules.

---

## 📊 Sample Insight

> **Rule**:  
> `IF (Product_High, ProdDur_Long) → THEN Purchase`  
> - **Confidence**: 48%  
> - **Lift**: 3.1  
>
> **Recommendation**:  
> Trigger an exit-intent popup with a 10% discount offer for users who browse many products for a long duration.

---

## 🖥️ Demo

Try the live app (if deployed):  
👉 [https://shopmind.streamlit.app](https://shopmind.streamlit.app)

Or run locally:
```bash
streamlit run app.py
```

![App Screenshot](screenshots/app_screenshot.png)  
*Example: User inputs session behavior and receives targeted recommendations.*

---

## 🧰 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/shopmind-recommender.git
cd shopmind-recommender
```

### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate shopmind-recommender
```

### 3. Launch the App
```bash
streamlit run app.py
```

> 📝 **Note**: Ensure all `.pkl` model files (`recommendation_rules.pkl`, `transaction_encoder.pkl`) are present.

---

## 📁 Project Structure

```
shopmind-recommender/
│
├── notebook.ipynb               # Full analysis and model training
├── app.py                       # Streamlit web application
├── environment.yml              # Conda environment definition
├── requirements.txt             # Pip fallback dependencies
├── recommendation_rules.pkl     # Trained association rules
├── transaction_encoder.pkl      # Encoder for input transformation
├── data/                        # Optional: local dataset storage
└── screenshots/                 # App visuals
```

---

## 📚 Dependencies

Built with:
- Python 3.10
- pandas, numpy
- matplotlib, seaborn
- mlxtend (for Apriori)
- scikit-learn
- streamlit

See `environment.yml` for full list.

---

## 🌐 Future Enhancements

- Incorporate **session sequence modeling** using Markov chains or RNNs.
- Integrate with real-time analytics platforms (e.g., Google Analytics).
- Add **SHAP-based explainability** for rule confidence.
- Support **A/B testing** of recommended interventions.

---

## 🙌 Acknowledgments

- Dataset: [UCI ML Repository - Online Shoppers Intention](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
- Libraries: `mlxtend`, `streamlit`, `pandas`

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

> Developed by [Elvis Tile](https://github.com/elvis07jr) | 2025  
> A data science project combining behavioral analytics and e-commerce optimization.

---
