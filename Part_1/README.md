# Crop Yield Prediction Using Machine Learning

## 🌱 Project Overview
This project aims to **predict crop yields** based on farming practices (e.g., irrigation, fertilizer use) and environmental factors (e.g., soil type, season) using machine learning. It helps farmers and policymakers optimize resource allocation and improve agricultural sustainability.

---

## 📌 Key Objectives
1. **Predict Crop Yield**: Accurately forecast yield (in tons) for different crops.
2. **Identify Key Factors**: Determine which variables (e.g., water usage, farm area) most impact yield.
3. **Support Decision-Making**: Provide actionable insights for farmers to reduce costs and improve productivity.

---

## 🛠️ Technical Components
### 1. **Data Preprocessing**
- **Handled Missing Values**: Imputed numerical/categorical data.
- **Feature Engineering**:  
  - Scaled numerical features (e.g., `Farm_Area`).  
  - Encoded categorical variables (e.g., `Crop_Type`, `Soil_Type`).  
- **Train-Validation-Test Split**: 70%-15%-15%.

### 2. **Model Development**
- **Algorithm**: Random Forest Regressor (chosen for handling mixed data types and robustness to overfitting).  
- **Hyperparameter Tuning**: Optimized `n_estimators` and `max_depth` using `GridSearchCV`.

### 3. **Evaluation Metrics**
- **R² Score**: Measures variance explained by the model (target: close to 1).  
- **RMSE**: Quantifies prediction errors in yield units (tons).  

### 4. **Deployment & Monitoring**
- **API**: Flask app for real-time predictions (e.g., `POST /predict`).  
- **Concept Drift Detection**: Kolmogorov-Smirnov test to monitor data distribution shifts over time.  

---

## 📂 Dataset Features
| Feature                | Description                          | Example Values              |
|------------------------|--------------------------------------|-----------------------------|
| `Crop_Type`            | Type of crop cultivated              | Cotton, Rice, Tomato        |
| `Irrigation_T`         | Irrigation method used               | Drip, Sprinkler, Flood      |
| `Soil_Type`            | Soil classification                  | Loamy, Clay, Sandy          |
| `Water_Usage`          | Water consumed (cubic meters)        | 76648.2, 68725.54           |
| **Target**: `Yield(tons)` | Crop output in tons               | 14.44, 42.91                |

---

## 🚀 Business Impact
- **Farmers**: Optimize water/fertilizer use to maximize yield.  
- **Policymakers**: Design subsidies based on predictive insights.  
- **Sustainability**: Reduce resource waste by aligning practices with model recommendations.  

---

## 🔄 Next Steps
1. **Expand Data Sources**: Integrate weather APIs for real-time climate data.  
2. **Deploy Cloud Solution**: Use AWS SageMaker for scalable predictions.  
3. **Farmer UI**: Develop a mobile app for easy access to yield forecasts.  

---

