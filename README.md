🌍 Globle CO₂ Emissions Analysis, Classification & Forecasting

A complete Machine Learning + Time-Series Project with Streamlit Dashboard & Global CO₂ Mapping

1. Project Overview

Climate change is driven largely by CO₂ emissions. Understanding which countries emit the most, how emissions change over time, and how well we can predict future emissions is key for policy-making, sustainability, and climate research.

This project performs:

 - Data cleaning & exploration
 - Global CO₂ trend analysis
 - Classification (high vs low emitters) using ML models
 - Forecasting future CO₂ levels using Facebook Prophet
 - Forecast validation using real OWID data
 - Global CO₂ Mapping (Plotly Choropleth + Time-lapse)
 - Interactive Streamlit Dashboard


2. Repository Structure


 CO2-Emissions-Analysis/
 │
 ├── data/
 ├── models/
 │     ├── Tuned_rf_model.pkl
 │     ├── Tuned_svm_model.pkl
 │     ├── Tuned_ann_model.pkl
 │     ├── encoder.pkl
 │
 ├── notebooks/
 │     ├── 01_Data_Cleaning_Exploration.ipynb
 │     ├── 02_Classification_Models.ipynb
 │     ├── 03_Forecasting_Prophet.ipynb
 │     └── 04_Model_Validation.ipynb
 │
 ├── app/
 │     └── streamlit_app.py
 │
 ├── visuals/
 │     ├── actual_vs_forecast.png
 │     ├── residuals.png
 │     └── model_metrics.png
 │
 ├── requirements.txt
 └── README.md


3. Dataset Description

 - Country -	Name of country
 - Region	- Continent
 - Date	  - Year of measurement
 - Kilotons of CO2	- Total annual emissions
 - Metric Tons Per Capita	- Per-person emissions

4. Tools & Technologies Used

1. Platform & Environment
 - Google Colab (main development notebook)
 - Streamlit (interactive CO₂ Forecast Validator & CO₂ Classification Dashboard)

2. Programming Language
 - Python 3.10+

3. Core Libraries for Data Analysis
 - Pandas – data manipulation and preprocessing
 - NumPy – numerical operations
 - Matplotlib & Seaborn – static visualizations and statistical plots
 - Plotly Express – interactive maps, charts, animated CO₂ choropleths

4. Machine Learning & Modeling

Classification Models
 - Scikit-learn
 - Random Forest Classifier
 - Support Vector Machine (SVM)
 - Artificial Neural Network (MLPClassifier)
 - GridSearchCV for hyperparameter tuning
 - Feature Engineering (OneHotEncoder, train_test_split)
 - Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, Cross-Validation

Time-Series Forecasting Models
 - Facebook Prophet
 - Forecast modeling
 - Trend decomposition
 - Future CO₂ prediction

5. Deployment & Dashboards

Streamlit: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://co2-emissions-analysis-j2cflxie7hcyvh2fqbjmrp.streamlit.app/)

 - CO₂ Emission Classification UI
 - CO₂ Forecast Validator (Prophet vs Actual OWID data)
 - Global CO₂ Map Visualization
 - CO₂ Top Emitters Ranking
 - Animated CO₂ Timeline (Plotly Choropleth)

GIF Preview of the Streamlit App :

![Streamlit_app_gif](https://github.com/user-attachments/assets/c383fe9c-9af1-4812-8012-48042951d344)




7. Data Sources
 - Kaggle Data (primary dataset – historical CO₂ emissions)
 - Our World In Data CO₂ Dataset (OWID) – real-world validation dataset (till 2023/2024)




