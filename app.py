from flask import Flask, render_template

app = Flask(__name__)

# HOME
@app.route("/")
def home():
    return render_template("index.html")

# PLANTEAMIENTO
@app.route("/planteamiento")
def planteamiento():
    return render_template("planteamiento.html")

# INGENIERIA DE DATOS
@app.route("/datos")
def datos():
    return render_template("datos.html")

if __name__ == "__main__":
    app.run(debug=True)