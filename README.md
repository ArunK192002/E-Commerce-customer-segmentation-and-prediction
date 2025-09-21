# 📊 E-commerce Customer Segmentation and Classification

## 📁 Project Overview

This project focuses on **customer segmentation and predictive modeling** for an e-commerce platform. The workflow includes:

- Data cleaning and feature engineering
- Customer segmentation using **KMeans, Agglomerative, and DBSCAN**
- Classification using **SVM, KNN, Random Forest, Gradient Boosting**
- Hyperparameter tuning with **Grid Search**
- Visualizations and business recommendations

---

## 📂 Project Structure

```text
🔼
├── ml_env/                           # Virtual environment folder
├── Visualization/                    # Folder containing clustering and feature plots
├── customer_segmentation.xlsx        # Processed customer segmentation data
├── data.xlsx                         # Raw e-commerce transactional data
├── Ecommerce_Customer_Segmentation_Project.ipynb  # Jupyter Notebook (Full analysis)
├── ecommerce_segmentation_app.py     # Python file (potential dashboard/app script)
├── label_encoder_country.pkl         # Saved Label Encoder for country
├── label_encoder_segment.pkl         # Saved Label Encoder for customer segments
├── requirements.txt                  # Required Python libraries
├── tuned_gradient_boosting_model.pkl # Final trained model (Gradient Boosting)
```

---

## ⚙️ Project Workflow

### 1. **Data Preprocessing**

- Loaded and cleaned the raw e-commerce data.
- Created features: Recency, Frequency, Monetary Value, Return Rate, Reorder Rate, Customer Lifetime, Basket Size, Average Order Value, Product Variety.
- Treated outliers using **IQR capping.**

### 2. **Customer Segmentation**

- Clustering using:
  - **KMeans (5 clusters)**
  - **Agglomerative Clustering**
  - **DBSCAN**
- Visualized clusters using **PCA.**
- Analyzed cluster sizes and segment profiles.

### 3. **Classification Modeling**

- Built and evaluated:
  - SVM
  - KNN
  - Random Forest
  - Gradient Boosting
- Hyperparameter tuning using **GridSearchCV.**
- Final model: **Tuned Gradient Boosting Classifier.**

### 4. **Model Deployment**

- Saved the tuned model and label encoders for future predictions.

---

## 🚀 How to Run the Project

### 1. **Set Up Virtual Environment**

```bash
python -m venv ml_env
source ml_env/Scripts/activate  # Windows
```

### 2. **Install Required Libraries**

```bash
pip install -r requirements.txt
```

### 3. **Run the Jupyter Notebook**

Open `Ecommerce_Customer_Segmentation_Project.ipynb` and run all cells.

### 4. **Run the App (Optional)**

If `ecommerce_segmentation_app.py` is a dashboard script, you can run it with:

```bash
python ecommerce_segmentation_app.py
```

### 5. **Load Saved Model for Predictions**

```python
import joblib

# Load Model and Encoders
model = joblib.load('tuned_gradient_boosting_model.pkl')
country_encoder = joblib.load('label_encoder_country.pkl')
segment_encoder = joblib.load('label_encoder_segment.pkl')
```

---

## 📈 Business Insights & Recommendations

- **Premium Loyalists:** Retain with loyalty programs, exclusive deals, and early access to new products.
- **High-Value Frequent Buyers:** Engage with personalized offers and premium customer support.
- **Infrequent Big Spenders:** Encourage more frequent purchases via targeted bundles and VIP benefits.
- **Moderate Shoppers:** Provide product recommendations and small incentives to boost frequency.
- **Churned/One-Time Customers:** Run re-engagement campaigns with special offers and reminders.

---

## 📅 Future Scope

- Integrate website browsing and campaign data for deeper segmentation.
- Build a real-time customer segmentation dashboard.
- Explore deep learning models for further performance improvements.

---

## ✍️ Author

Arun 

---

