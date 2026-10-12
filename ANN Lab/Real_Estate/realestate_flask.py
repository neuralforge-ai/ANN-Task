import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ---- Train model (house price per unit area) ----
df = pd.read_excel("RealEstate.xlsx")
X = df.iloc[:, 1:7]          # skip the "No" index column
y = df.iloc[:, 7]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

FIELDS = [
    ("f0", "Transaction Date"), ("f1", "House Age"),
    ("f2", "Distance to nearest MRT (m)"), ("f3", "Number of Convenience Stores"),
    ("f4", "Latitude"), ("f5", "Longitude"),
]

@app.route("/", methods=["GET", "POST"])
def home():
    result, error, values = None, None, {}
    if request.method == "POST":
        try:
            values = {key: request.form[key] for key, _ in FIELDS}
            row = pd.DataFrame([[float(values[k]) for k, _ in FIELDS]], columns=X.columns)
            result = f"House Price (per unit area) = {model.predict(row)[0]:.2f}"
        except (KeyError, ValueError):
            error = "Please fill every field with a valid number."
    return render_template("form.html", title="Real Estate Price Prediction",
                           fields=FIELDS, values=values, result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True, port=5001)   # python realestate_flask.py
