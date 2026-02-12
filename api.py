from flask import Flask, request, jsonify, render_template
from graph import graph

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze-portfolio", methods=["POST"])
def analyze():
    data = request.json

    result = graph.invoke({
        "user_portfolio": data["user_portfolio"],
        "user_needs": data.get("user_needs", {})
    })

    return jsonify({
        "stable_performers": result.get("stable_performers"),
        "potential_reducers": result.get("potential_reducers"),
        "evaluation": result.get("eval_output"),
        "allocation_plan": result.get("allocation_plan")
    })

if __name__ == "__main__":
    app.run(debug=True)
