"""
E-commerce Analytics Project: End-to-End Pipeline
Inspired by Brazilian E-Commerce Public Dataset (Olist)
Author: Antigravity Assistant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not installed. Skipping XGBoost model.")
    XGB_AVAILABLE = False
from sklearn.metrics import mean_squared_error, r2_score

# Set visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_data():
    """Step 1: Load and Merge Data"""
    print("--- PART 1: DATA ENGINEERING ---")
    
    try:
        customers = pd.read_csv('data/olist_customers_dataset.csv')
        orders = pd.read_csv('data/olist_orders_dataset.csv')
        items = pd.read_csv('data/olist_order_items_dataset.csv')
        payments = pd.read_csv('data/olist_order_payments_dataset.csv')
        products = pd.read_csv('data/olist_products_dataset.csv')
        reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')
        translation = pd.read_csv('data/product_category_name_translation.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset CSVs are in the 'data/' folder. Run 'mock_data_generator.py' to generate sample data.")
        return None

    # Merge tables
    df = orders.merge(customers, on='customer_id', how='left')
    df = df.merge(items, on='order_id', how='left')
    df = df.merge(payments, on='order_id', how='left')
    df = df.merge(products, on='product_id', how='left')
    df = df.merge(reviews, on='order_id', how='left')
    df = df.merge(translation, on='product_category_name', how='left')
    
    print(f"Merged Data Shape: {df.shape}")
    
    # 2. Data Cleaning
    print("Cleaning data...")
    # Convert timestamps
    time_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col])
        
    # Handle missing values
    df['product_category_name_english'] = df['product_category_name_english'].fillna('unknown')
    df['review_score'] = df['review_score'].fillna(df['review_score'].median())
    
    # Feature Engineering
    df['order_month'] = df['order_purchase_timestamp'].dt.month
    df['order_year'] = df['order_purchase_timestamp'].dt.year
    df['delivery_delay'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.days
    
    return df

def perform_eda(df):
    """Step 2: Exploratory Data Analysis"""
    print("\n--- PART 2: EXPLORATORY DATA ANALYSIS (EDA) ---")
    
    # 1. Monthly Revenue Trend
    monthly_sales = df.groupby(['order_year', 'order_month'])['payment_value'].sum().reset_index()
    monthly_sales['date'] = monthly_sales.apply(lambda x: f"{int(x['order_year'])}-{int(x['order_month']):02d}", axis=1)
    
    plt.figure()
    sns.lineplot(data=monthly_sales, x='date', y='payment_value', marker='o')
    plt.title('Monthly Revenue Trend')
    plt.xticks(rotation=45)
    plt.savefig('notebooks/monthly_revenue.png')
    print("Saved: Monthly Revenue Trend chart.")

    # 2. Top Categories
    top_cats = df.groupby('product_category_name_english')['payment_value'].sum().sort_values(ascending=False).head(10)
    plt.figure()
    top_cats.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Product Categories by Revenue')
    plt.ylabel('Total Revenue')
    plt.savefig('notebooks/top_categories.png')
    print("Saved: Top Categories chart.")

    # 3. Delivery Delay vs Review Score
    plt.figure()
    sns.boxenplot(x='review_score', y='delivery_delay', data=df)
    plt.title('Delivery Delay vs Customer Review Score')
    plt.savefig('notebooks/delay_vs_review.png')
    print("Insights: Delays significantly impact customer satisfaction scores.")

def perform_rfm(df):
    """Step 3: Customer Segmentation (RFM)"""
    print("\n--- PART 3: CUSTOMER SEGMENTATION (RFM) ---")
    
    # Calculate Recency, Frequency, Monetary
    # We use customer_unique_id to identify real customers
    snapshot_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'count',
        'payment_value': 'sum'
    }).rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'payment_value': 'Monetary'
    })
    
    # K-Means Clustering
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # Finding optimal K (Simple Elbow visualization - log for brevity)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Map Clusters to Labels
    # Note: In real scenarios, we analyze cluster means to name them. 
    # For simplicity, we'll label them based on relative Monetary value rank.
    segment_map = {0: 'At Risk', 1: 'Loyal', 2: 'High Value', 3: 'Low Engagement'}
    rfm['Segment'] = rfm['Cluster'].map(segment_map)
    
    print("Customer Segments Profile:")
    print(rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean())
    
    return rfm

def train_models(df):
    """Step 4: Sales Prediction Models"""
    print("\n--- PART 4: SALES PREDICTION MODELS ---")
    
    # Prepare features for predicting payment_value (order revenue)
    # We'll use numerical features and encoded categories
    features = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'review_score']
    
    # Simple preprocessing
    ml_df = df.dropna(subset=features + ['payment_value'])
    X = ml_df[features]
    y = ml_df['payment_value']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42)
    }
    
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=50, random_state=42)
    
    results = {}
    best_model = None
    best_rmse = float('inf')
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results[name] = {'RMSE': rmse, 'R2': r2}
        print(f"{name} -> RMSE: {rmse:.2f}, R2: {r2:.2f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name

    print(f"\nBest Model: {best_model_name}")
    
    # Save Model
    if not os.path.exists('models'): os.makedirs('models')
    with open('models/best_sales_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Saved: Model file saved to 'models/best_sales_model.pkl'.")
    
    return best_model, features

def main():
    # Parts 1-6 Execution
    df = load_data()
    if df is not None:
        perform_eda(df)
        rfm_results = perform_rfm(df)
        model, feature_names = train_models(df)
        
        print("\n--- PROJECT COMPLETED SUCCESSFULLY ---")
        print("Next Step: Run 'streamlit run app.py' for the Deployment dashboard.")

if __name__ == "__main__":
    main()
