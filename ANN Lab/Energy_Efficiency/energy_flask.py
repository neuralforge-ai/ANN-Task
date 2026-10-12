import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ---- Train model (Heating Load = Y1) ----
df = pd.read_excel("EnergyEfficiency.xlsx")
X = df.iloc[:, 0:8]
y = df["Y1"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

FIELDS = [
    ("f0", "Relative Compactness"), ("f1", "Surface Area"),
    ("f2", "Wall Area"),            ("f3", "Roof Area"),
    ("f4", "Overall Height"),       ("f5", "Orientation"),
    ("f6", "Glazing Area"),         ("f7", "Glazing Distribution"),
]

@app.route("/", methods=["GET", "POST"])
def home():
    result, error, values = None, None, {}
    if request.method == "POST":
        try:
            values = {key: request.form[key] for key, _ in FIELDS}
            row = pd.DataFrame([[float(values[k]) for k, _ in FIELDS]], columns=X.columns)
            result = f"Heating Load = {model.predict(row)[0]:.2f}"
        except (KeyError, ValueError):
            error = "Please fill every field with a valid number."
    return render_template("form.html", title="Energy Efficiency Prediction",
                           fields=FIELDS, values=values, result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)   # run from terminal: python energy_flask.py
