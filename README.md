# 🏠 House Rent Prediction using Machine Learning

## 📘 Overview
This project predicts **house rent prices in Hyderabad** using **Linear Regression**.  
It leverages Python’s powerful data science libraries like `pandas`, `numpy`, `scikit-learn`, and `matplotlib` for model building, and **Streamlit** for creating an interactive web app.



## 🎯 Objective
To predict the **rent price of a house** based on its:
- Area (in square feet)
- Number of Bedrooms
- Number of Washrooms



## 🧠 Machine Learning Workflow

### 1️⃣ Data Collection
- Dataset: **Hyderabad_House_Data.csv**
- The dataset contains housing details such as *Price*, *Area*, *Bedrooms*, and *Washrooms*.

### 2️⃣ Data Preprocessing
- Handle missing values using `fillna()`
- Convert columns to numeric using `pd.to_numeric()`
- Clean and extract numerical values from textual data (like area)
- Encode categorical columns using `pd.get_dummies()`

### 3️⃣ Model Building
- **Algorithm:** Linear Regression  
- **Library:** `scikit-learn`
- Data split using `train_test_split()`
- Scaled features using `StandardScaler`

### 4️⃣ Model Evaluation
Metrics used:
- **MAE (Mean Absolute Error)**
- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

### 5️⃣ Model Saving
- Model saved using **Joblib**:
  - `house_rent_linear_model.pkl`
  - `scaler.pkl`



## 💻 Streamlit Web App

An interactive interface built using **Streamlit** where users can input:
- Area (in sq. ft.)
- Bedroom Type (e.g., 2 BHK Apartment)
- Number of Washrooms

and instantly get the **predicted house rent price**.

### 🔹 Run the App Locally

streamlit run app.py


---

## 📂 Project Structure
```
House-Rent-Prediction/
│
├── app.py                         # Streamlit web application
├── house_rent_linear_model.pkl    # Saved trained model
├── scaler.pkl                     # Saved scaler
├── Hyderabad_House_Data.csv       # Dataset
├── house_rent_prediction.ipynb    # Jupyter Notebook / Python training file
└── README.md                      # Project documentation
```

---

## ⚙️ Technologies Used
| Category | Tools / Libraries |
|-----------|------------------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deployment | Streamlit |
| Model Persistence | Joblib |


## 📈 Results
- The Linear Regression model provides a reliable estimate of house rent prices based on the input features.
- Example output:
  
  Predicted House Rent Price: ₹35,000.00
  



## ✨ Future Improvements
- Integrate advanced models like Random Forest or XGBoost
- Add more city datasets
- Build a complete dashboard for price comparison



## 🙋‍♀️ Author
**Thota Anushka**  
📧 Email: [anushkathota85@gmail.com](mailto:anushkathota85@gmail.com)  
💼 [LinkedIn Profile](https://www.linkedin.com/in/anushka-thota-3abb04384)



⭐ *If you found this project useful, don’t forget to give it a star on GitHub!*
