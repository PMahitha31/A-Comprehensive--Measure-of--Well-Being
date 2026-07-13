# 🌍 HDI Predictor — Human Development Index Estimator

A production-ready, end-to-end Machine Learning web application that predicts a country's (or hypothetical region's) **Human Development Index (HDI)** score and classifies it into a development tier — **Very High**, **High**, **Medium**, or **Low** — based on three core indicators:

- **Life Expectancy at Birth**
- **Mean Years of Schooling**
- **GNI Per Capita**

Built with a Linear Regression model trained using Scikit-learn and served through a Flask web application with a clean, modern UI.

---

## ✨ Features

- Clean, responsive web form for entering development indicators
- Real-time HDI score prediction using a trained Linear Regression model
- Automatic classification into UNDP-style development tiers
- Input validation with clear error messaging
- Exploratory Data Analysis (EDA) visuals: correlation heatmap and feature scatter plots
- Modern, professional UI (blue/slate/white palette)
- Modular, well-documented Python codebase ready for extension

---

## 🛠️ Technical Stack

| Layer            | Technology                          |
|-------------------|--------------------------------------|
| Language          | Python 3.9+                          |
| Web Framework     | Flask                                 |
| Machine Learning  | Scikit-learn (Linear Regression)      |
| Data Handling     | Pandas, NumPy                         |
| Visualization     | Matplotlib, Seaborn                   |
| Frontend          | HTML5, CSS3                           |
| Model Persistence | Pickle                                |

---

## 📁 Project Structure

```
hdi_predictor/
│
├── static/
│   └── css/
│       ├── style.css
│       ├── correlation_heatmap.png     (generated after training)
│       └── feature_scatterplots.png    (generated after training)
├── templates/
│   ├── index.html
│   └── result.html
├── data/
│   └── hdi_dataset.csv
├── app.py
├── train_model.py
├── hdi_model.pkl        (generated after training)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Extract the project

Unzip the downloaded archive to a folder of your choice, then navigate into it:

```bash
cd hdi_predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

This step loads `data/hdi_dataset.csv`, performs preprocessing and EDA, trains the Linear Regression model, evaluates it, and saves `hdi_model.pkl`.

```bash
python train_model.py
```

Expected console output includes the dataset shape, R² score, MAE, RMSE, and model coefficients. Two visualization files will also be created inside `static/css/`.

### 5. Run the Flask application

```bash
python app.py
```

The app will start at:

```
http://127.0.0.1:5000/
```

### 6. Use the application

1. Open the URL above in your browser.
2. Enter values for **Life Expectancy**, **Mean Years of Schooling**, and **GNI Per Capita**.
3. Click **Predict HDI Score**.
4. View the predicted HDI score along with its development category badge.

---

## 📊 Model Details

- **Algorithm:** Linear Regression (`sklearn.linear_model.LinearRegression`)
- **Features:** `Life_Expectancy`, `Mean_Years_Schooling`, `GNI_Per_Capita`
- **Target:** `HDI_Score`
- **Preprocessing:** Mean imputation for any missing numeric values
- **Evaluation Metrics:** R², MAE, RMSE (printed during training)

### HDI Classification Thresholds

| Category    | Score Range     |
|-------------|-----------------|
| Very High   | ≥ 0.800          |
| High        | 0.700 – 0.799    |
| Medium      | 0.550 – 0.699    |
| Low         | < 0.550          |

---

## 🔄 Retraining with New Data

To retrain the model with a larger or updated dataset:

1. Replace or append rows to `data/hdi_dataset.csv`, keeping the same column names:
   `Country, Life_Expectancy, Mean_Years_Schooling, GNI_Per_Capita, HDI_Score`
2. Re-run:
   ```bash
   python train_model.py
   ```
3. Restart the Flask app to load the newly saved `hdi_model.pkl`.

---

## 🧪 Notes & Limitations

- The bundled dataset is a small, illustrative sample (~20 rows) intended for demonstration. For production accuracy, use the full UNDP Human Development Report dataset.
- Predictions are clipped to the valid HDI range of `[0, 1]`.
- Linear Regression assumes a linear relationship between indicators and HDI; consider polynomial or ensemble models for higher fidelity on larger datasets.

---

## 📄 License

This project is provided as an educational/demonstration template and may be freely modified and reused.
