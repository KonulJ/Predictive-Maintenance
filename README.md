# 🛠️ Predictive Maintenance for Aircraft Turbofan Engines using Machine Learning  
**Powered by the C-MAPSS Dataset**

---

## 📌 Overview  
This project demonstrates how machine learning (especially regression) models can be applied to sensor and operational settings data from jet engines to predict **Remaining Useful Life (RUL)** and prevent equipment failure. By leveraging the **C-MAPSS dataset**, the application predicts potential breakdowns before they occur—reducing downtime, optimizing maintenance schedules, and improving operational efficiency.

An **interactive dashboard** built using **Plotly Dash** provides visual insights into sensor behavior and model performance.

---

## 📂 Project Structure  

├── dataset/                # Raw and processed CSV files (sensor data, labels)  
├── models/                 # Saved model files (optional)  
├── app.py                  # Main Dash application  
├── requirements.txt        # Required Python packages  
├── README.md               # You're here :) '''

---

## 📊 Dataset: C-MAPSS  
Provided by **NASA** for engine degradation simulation.  

Each engine undergoes a run-to-failure simulation under different conditions.  

**Dataset contains:**  
- "engine_id", "time_in_cycles" 
- 3 operational settings  
- 21 sensor readings  
- Ground truth **Remaining Useful Life (RUL)** in a separate file for the test set  

---

## ⚙️ Feature Engineering

- Applied **Min-Max Normalization** and **"first-n-cycle" normalization**
- Created smoothed **rolling averages** for sensor signals
- **RUL clipping** to handle extreme values and reduce noise

---

## 🤖 Models  
Multiple regressors were tested:  
- ✅ **XGBoost**  
- ✅ **LightGBM**  

**Hyperparameter Tuning:**  
- Used **GridSearchCV** for optimization  

**Evaluation Metrics:**  
- **MAE**, **MSE**, **RMSE**, **R² Score**  

---

## 📈 Dashboard Features  
- 📌 **Sensor behavior vs RUL** for selected engine  
- 📉 **Model comparison bar charts**  
- 📘 **Data exploration and preprocessing steps explained**  
- 📦 **Modular design** using Plotly Dash  

---

## 💡 Business Value  
- Reduces unexpected breakdowns  
- Enables **condition-based maintenance**  
- Saves operational costs and increases asset longevity  
- Offers a scalable solution for industrial applications  

---

## 🧩 Challenges  
- Accuracy (~90%) leaves room for improvement  
- Predictive power can be improved with **classification-based early warning** (e.g., risk zone detection within 30 cycles of failure)  
- Sensor noise and real-time deployment challenges  

---

## 🚀 Future Work  
- Integrate **classification models** for risk zones  
- Deploy the dashboard using **Docker / Heroku / Streamlit Sharing**  
- Use **LSTM** for sequence modeling and time-dependent features
---

# Clone the repo
git clone git clone https://github.com/KonulJ/Predictive-Maintenance.git
cd predictive-maintenance

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
