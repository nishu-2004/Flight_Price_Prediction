---

#  Flight Price Prediction using ANN  

This project predicts flight ticket prices using an **Artificial Neural Network (ANN)**. The model is trained on a dataset containing flight details such as airline, source, destination, duration, and date of journey.  

---

##  Dataset  

The dataset used for this project:  
 **[Flight Price Prediction Dataset](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)**  

It includes features like:  
✔️ Airline  
✔️ Source & Destination  
✔️ Date of Journey  
✔️ Duration  
✔️ Total Stops  

These features help predict **ticket prices** accurately.  

---

##  Methodology  

🔹 **Data Preprocessing** – Handling missing values, feature encoding, and normalization.  
🔹 **Feature Engineering** – Extracting relevant information like day, month, and total stops.  
🔹 **ANN Model** – A deep learning model trained using TensorFlow/Keras.  
🔹 **Evaluation Metrics** – Performance measured using **Mean Absolute Error (MAE)**.  

---

##  Results  

🔹 **MAE (Log-Transformed Scale):** 0.1372  
🔹 **MAE (Original Scale):** $2322.08  

---

##  Technologies Used  

- **Python**  
- **TensorFlow/Keras**  
- **Pandas, NumPy, Matplotlib**  

---

##  Usage  

### Clone the repository  
```bash
git clone https://github.com/yourusername/flight-price-prediction.git
cd flight-price-prediction
```

### Install dependencies  
```bash
pip install -r requirements.txt
```

### Train the model  
```bash
python flight_model.py
```
🔗 GitHub: nishu-2004(https://github.com/nishu-2004)  

