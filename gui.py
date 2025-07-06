import sys,os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QComboBox,
    QPushButton, QVBoxLayout, QFormLayout, QMessageBox, QFrame
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import joblib

DAY_COUNT_RANGE = (1, 1100)
CALORIE_RANGE = (50, 2000)
INFLATION_RANGE = (0, 30)
UNEMPLOYMENT_RANGE = (0, 30)

def resource_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath("."), filename)

model = joblib.load(resource_path("model.pkl"))
le_specials = joblib.load(resource_path("le_specials.pkl"))
le_mods = joblib.load(resource_path("le_mods.pkl"))

with open(resource_path("one_hot_columns.txt")) as f:
    one_hot_columns = f.read().splitlines()

class SalesPredictorGUI(QWidget):
    def __init__(self, model, le_specials, le_mods, one_hot_columns):
        super().__init__()
        self.model = model
        self.le_specials = le_specials
        self.le_mods = le_mods
        self.one_hot_columns = one_hot_columns
        self.setWindowTitle("Sales Predictor")
        self.setFixedWidth(650)
        self.inputs = {}
        self.setup_ui()
        self.setStyleSheet(self.load_styles())

    def setup_ui(self):
        layout = QVBoxLayout()
        form_layout = QFormLayout()

        fields = [
            ("Day_Count", QLineEdit, "1–1100"),
            ("Day", QComboBox, ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']),
            ("ProductName", QComboBox, ['VBurger','No Chz','Veg Burger', 'ClassicWrap', 
                                        'Fries','Cheese Burger','Chicken Burger','Carmalized Onions','No Tom Plz','Chicken Br',
                                        'Chicken Brg','Chk Burger','Chz Burger','Coca-Cola','Coke','Extra Cheese','F Burger','Falafel Burger',
                                        'Fast Please','NO TOMATOES','No Lettuce','Veggie Burger','X-CHeese','Absolute no cheese']),
            ("Product_Calorie", QLineEdit, "50–2000"),
            ("specials", QComboBox, ['No', 'Yes']),
            ("mods", QComboBox, ['No', 'Y']),
            ("weather", QComboBox, ['hot', 'cold', 'rainy', 'very hot', 'humid', 'very cold']),
            ("Inflation_Percentage", QLineEdit, "0–30%"),
            ("Unemployment_Percentage", QLineEdit, "0–30%")
        ]

        for name, widget_type, *args in fields:
            widget = widget_type()
            if widget_type == QComboBox:
                widget.addItems(args[0])
            else:
                widget.setPlaceholderText(args[0])
            self.inputs[name] = widget
            form_layout.addRow(f"{name}:", widget)

        layout.addLayout(form_layout)

        predict_btn = QPushButton("Predict Total Sale")
        predict_btn.clicked.connect(self.predict_sales)
        predict_btn.setCursor(Qt.PointingHandCursor)
        layout.addWidget(predict_btn)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        self.result_label = QLabel("Prediction: ₹0.00")
        self.result_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def load_styles(self):
        return """
        QWidget {
            background-color: #f4f4f4;
            font-family: 'Segoe UI';
            font-size: 11pt;
        }
        QLabel {
            font-weight: 500;
        }
        QLineEdit, QComboBox {
            padding: 6px;
            border: 1px solid #aaa;
            border-radius: 4px;
        }
        QPushButton {
            background-color: #007acc;
            color: white;
            padding: 8px;
            border: none;
            border-radius: 6px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #005fa3;
        }
        """

    def validate_input(self, field_name, value, dtype, min_val, max_val):
        try:
            typed_val = dtype(value)
            if not (min_val <= typed_val <= max_val):
                raise ValueError
            return typed_val
        except:
            raise ValueError(f"{field_name} must be a {dtype.__name__} between {min_val} and {max_val}.")

    def predict_sales(self):
        try:
            required_fields = {
                'Day_Count': (int, *DAY_COUNT_RANGE),
                'Product_Calorie': (int, *CALORIE_RANGE),
                'Inflation_Percentage': (float, *INFLATION_RANGE),
                'Unemployment_Percentage': (float, *UNEMPLOYMENT_RANGE)
            }

            input_values = {}
            for field, (dtype, min_val, max_val) in required_fields.items():
                raw_value = self.inputs[field].text().strip()
                if not raw_value:
                    raise ValueError(f"{field} cannot be empty.")
                input_values[field] = self.validate_input(field, raw_value, dtype, min_val, max_val)

            input_values.update({
                'Day': self.inputs['Day'].currentText(),
                'ProductName': self.inputs['ProductName'].currentText(),
                'specials': self.inputs['specials'].currentText(),
                'mods': self.inputs['mods'].currentText(),
                'weather': self.inputs['weather'].currentText()
            })

            df = pd.DataFrame([input_values])

            df['specials'] = self.le_specials.transform(df['specials'])
            df['mods'] = self.le_mods.transform(df['mods'])

            df = pd.get_dummies(df, columns=['Day', 'ProductName', 'weather'])

            for col in self.one_hot_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.one_hot_columns]

            prediction = self.model.predict(df)[0]
            prediction = max(0, prediction)  
            self.result_label.setText(f"Prediction: ₹{prediction:,.2f}")

        except ValueError as ve:
            QMessageBox.warning(self, "Invalid Input", str(ve))
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = SalesPredictorGUI(model, le_specials, le_mods, one_hot_columns)
    window.show()
    sys.exit(app.exec_())

    

