from flask import Flask, render_template, request
from logistic import run_logistic, predict_risk
from rforest import run_forest, predict_forest
from lgbm_model import run_light, predict_light
from xgboost_model import run_xgboost, predict_xgboost

app = Flask(__name__)

# HOME
@app.route("/")
def home():
    return render_template("index.html")

# APPROACH
@app.route("/approach")
def approach():
    return render_template("approach.html")

# DATA
@app.route("/data")
def data():
    return render_template("data.html")

# =========================
# LOGISTIC
# =========================
@app.route("/logistic")
def logistic():
    datos = run_logistic()
    return render_template("logistic.html", datos=datos)

@app.route("/predict_logistic", methods=["POST"])
def predict_logistic():
    porcentaje_led = float(request.form["porcentaje_led"])
    total          = float(request.form["total"])
    resultado      = predict_risk(porcentaje_led, total)
    datos          = run_logistic()
    return render_template("logistic.html", datos=datos, prediction=resultado)

# =========================
# RANDOM FOREST
# =========================
@app.route("/forest")
def forest():
    datos = run_forest()
    return render_template("forest.html", datos=datos)

@app.route("/predict_forest", methods=["POST"])
def predict_forest_route():
    porcentaje_led = float(request.form["porcentaje_led"])
    total          = float(request.form["total"])
    resultado      = predict_forest(porcentaje_led, total)
    datos          = run_forest()
    return render_template("forest.html", datos=datos, prediction=resultado)

# =========================
# LIGHTGBM
# =========================
@app.route("/light")
def light():
    datos = run_light()
    return render_template("light.html", datos=datos)

@app.route("/predict_light", methods=["POST"])
def predict_light_route():
    porcentaje_led = float(request.form["porcentaje_led"])
    total          = float(request.form["total"])
    resultado      = predict_light(porcentaje_led, total)
    datos          = run_light()
    return render_template("light.html", datos=datos, prediction=resultado)

# =========================
# XGBOOST
# =========================
@app.route("/xgboost")
def xgboost():
    datos = run_xgboost()
    return render_template("xgboost.html", datos=datos)

@app.route("/predict_xgboost", methods=["POST"])
def predict_xgboost_route():
    porcentaje_led = float(request.form["porcentaje_led"])
    total          = float(request.form["total"])
    resultado      = predict_xgboost(porcentaje_led, total)
    datos          = run_xgboost()
    return render_template("xgboost.html", datos=datos, prediction=resultado)
#-----------------
#clustering
#-----------------
from clustering import run_clustering, predict_cluster

@app.route("/clustering")
def clustering():
    datos = run_clustering()
    return render_template("clustering.html", datos=datos)

@app.route("/predict_clustering", methods=["POST"])
def predict_clustering():
    porcentaje_led = float(request.form["porcentaje_led"])
    total          = float(request.form["total"])
    led            = float(request.form["led"])
    mes            = request.form["mes"]
    localidad      = request.form["localidad"]
    resultado      = predict_cluster(porcentaje_led, total, led, mes, localidad)
    datos          = run_clustering()
    return render_template("clustering.html", datos=datos, prediction=resultado)


# =========================
# EXPLANATIONS
# =========================
@app.route("/logistic_explanation")
def logistic_explanation():
    return render_template("logistic_explanation.html")

@app.route("/forest_explanation")
def forest_explanation():
    return render_template("forest_explanation.html")

@app.route("/light_explanation")
def light_explanation():
    return render_template("light_explanation.html")

@app.route("/xgboost_explanation")
def xgboost_explanation():
    return render_template("xgboost_explanation.html")

@app.route("/clustering_explanation")
def clustering_explanation():
    return render_template("clustering_explanation.html")

if __name__ == "__main__":
    app.run(debug=True)
