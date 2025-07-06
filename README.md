# Sales Prediction with PyQt GUI

This project builds a machine learning model that predicts **total combined sales** across 3 stores for each product using a synthetically generated dataset over 3 years.

## 📁 Files in the Repository

- `artificial_sales.csv` – Input dataset
- `model.py` – Script to train and save the model
- `model.pkl` – Trained model file
- `gui.py` – PyQt5 GUI for user interaction and predictions
- `le_mods.pkl`, `le_specials.pkl`, `one_hot_columns.txt` – Encoders and columns used during preprocessing
- `SalesPredictor.spec` – PyInstaller config (used to build `.exe`)
- `README.md` – Project documentation

## 🧠 ML Model

- Uses `RandomForestRegressor` and/or `DecisionTreeRegressor`
- Feature engineering includes label encoding and one-hot encoding
- Evaluated using R², RMSE, and custom accuracy

## 💻 GUI

Built using PyQt5. Allows users to enter product data and get sales prediction.

## ⚙️ Running the Project

### Installing the required dependencies
```bash
pip install pandas, numpy, scikit-learn, pyqt5, joblib, pyinstaller
```

### Train the model
```bash
python model.py
```

### Run the gui
```bash
python gui.py
```

### You can also build the executable binary using:
```bash
pyinstaller SalesPredictor.spec
```

