"""
Problem: Predict combined sales of all three stores for each revenue generating product.
Assumptions:
- Data contains columns: Store, ProductName, Amt, Day, and other features.
- We predict total sales per product per day (across all stores).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import warnings

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    if 'Promo_applied' in df.columns:
        df.drop(columns=['Promo_applied'], inplace=True)
    if 'Day' in df.columns:
        df['Day'].fillna(df['Day'].mode()[0], inplace=True)
    if 'Inflation_Percentage' in df.columns:
        df['Inflation_Percentage'].fillna(df['Inflation_Percentage'].median(), inplace=True)
    df.dropna(subset=['Amt'], inplace=True)
    return df

def aggregate_sales(df):
    group_cols = ['Day_Count', 'Day', 'ProductName', 'Product_Calorie', 'specials', 'mods', 'weather', 'Inflation_Percentage', 'Unemployment_Percentage']
    df_grouped = df.groupby(group_cols, as_index=False)['Amt'].sum()
    df_grouped.rename(columns={'Amt': 'Total_Sales'}, inplace=True)
    return df_grouped

def encode_features(df):
    le_specials = LabelEncoder()
    le_mods = LabelEncoder()
    df['specials'] = le_specials.fit_transform(df['specials'])
    df['mods'] = le_mods.fit_transform(df['mods'])

    df_encoded = pd.get_dummies(df, columns=['Day', 'ProductName', 'weather'], drop_first=False)

    joblib.dump(le_specials, 'le_specials.pkl')
    joblib.dump(le_mods, 'le_mods.pkl')

    with open("one_hot_columns.txt", "w") as f:
        f.writelines(col + "\n" for col in df_encoded.drop(columns=['Total_Sales']).columns)

    return df_encoded

def custom_accuracy(y_true, y_pred, threshold=45000):
    correct = sum(abs(y_t - y_p) <= threshold for y_t, y_p in zip(y_true, y_pred))
    return correct / len(y_true)

def evaluate_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    acc = custom_accuracy(y_test, y_pred)
    return r2, rmse, acc

def main():
    df = load_and_clean_data("artificial_sales.csv")
    df_grouped = aggregate_sales(df)
    df_encoded = encode_features(df_grouped)

    X = df_encoded.drop(columns=['Total_Sales'])
    y = df_encoded['Total_Sales']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    models = {
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    }

    print("Base Model Evaluation:")
    results = {}
    trained_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        r2, rmse, acc = evaluate_model(pipe, x_train, x_test, y_train, y_test)
        print(f"\n=== {name} ===")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: ₹{rmse:,.2f}")
        print(f"Custom Accuracy (±₹45,000): {acc * 100:.2f}%")
        results[name] = {"r2": r2, "rmse": rmse, "accuracy": acc}
        trained_pipelines[name] = pipe

    best_model_name = max(results, key=lambda k: results[k]["r2"])
    best_model = trained_pipelines[best_model_name]

    print(f"\nBest model: {best_model_name}")
    
    joblib.dump(best_model, "model.pkl")


if __name__ == "__main__":
    main()