from flask import Flask, render_template

#define app
app = Flask(__name__)

@app.route("/")
def mainScreen():
    return render_template("index.html")

@app.route("/setupCreation")
def setupCreation():
    return render_template("setupCreation.html")

if __name__ == "__main__":
    app.run(debug=True)
