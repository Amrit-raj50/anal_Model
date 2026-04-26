# 🛍️ E-Commerce Analytics & Sales Prediction

A complete end-to-end data science project based on the **Olist Brazilian E-Commerce Dataset**. This project transforms raw transactional data into actionable business insights using Machine Learning and Exploratory Data Analysis.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data (Optional)
If you don't have the 100MB+ Olist dataset yet, run this to create simulated data for testing:
```bash
python mock_data_generator.py
```

### 3. Run Analytics Pipeline
This script performs Data Engineering, EDA, RFM Analysis, and Model Training.
```bash
python ecommerce_analytics.py
```

### 4. Launch Dashboard
```bash
streamlit run app.py
```

---

## 📊 Project Highlights

### 1. Data Engineering
- **Merging**: Unified 7 relational tables (orders, customers, products, items, etc.) into a single master dataframe.
- **Cleaning**: Handled missing values in product categories and normalized timestamps for time-series analysis.
- **Feature Engineering**: Extracted 'Delivery Delay' and 'Order Month' to capture seasonal and operational patterns.

### 2. Customer Segmentation (RFM)
We categorized customers into four distinct business personas:
- **Champions/High Value**: Frequent buyers with high spend.
- **Loyal**: Consistent engagement, median spend.
- **At Risk**: Have not purchased recently; need re-engagement campaigns.
- **Low Engagement**: Single-time buyers with low order value.

### 3. Sales Prediction
We compared three powerful models to predict final order value:
- **Linear Regression**: Our baseline. High interpretability but struggles with non-linear relationships.
- **Random Forest**: Excellent at handling outlier data and complex interactions.
- **XGBoost**: Generally the top performer. It uses Gradient Boosting to iteratively correct errors, making it highly accurate for price prediction.

---

## 🌍 Deployment

To deploy this project for free on **Streamlit Cloud**:

1.  **Push to GitHub**: Create a repository and upload all files (including `data/` and `models/`).
2.  **Sign in to Streamlit Cloud**: Connect your GitHub account.
3.  **New App**: Select your `anal_Model` repository and point the main file to `app.py`.
4.  **Deploy**: Streamlit will automatically install dependencies and launch your dashboard at a public URL!

---

## 💡 Business Impact
...

## ⚖️ Ethical Considerations
- **Dataset Bias**: This data represents the Brazilian market (2016-2018). Trends may differ in the Indian (Myntra) or Global markets.
- **Privacy**: All customer names and contact details have been anonymized (using unique IDs) to protect user identity.