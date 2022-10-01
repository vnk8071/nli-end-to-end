import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import render_template, request, flash
from __init__ import create_app
from models import NLIDatabase
from database import db, drop_table
from bert.inference import bert_inference

app = create_app()

@app.before_first_request
def create_table():
    db.create_all(app=create_app())

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=(["GET", "POST"]))
def user_input():
    if request.method == "POST":
        if not request.form["premise"] or not request.form["hypothesis"]:
            flash("Please enter all fields", "error")
        else:
            user_premise = request.form["premise"]
            user_hypothesis = request.form["hypothesis"]
            user_model = request.form["model"]
            if user_model == "BERT":
                (premise, hypothesis), (index, mask, token_type), (prob, pred) = bert_inference(user_premise, user_hypothesis)
                print(f"Prob {prob}")
                print(f"Pred {pred}")
    return render_template("predict.html", user_premise=premise, user_hypothesis=hypothesis)
if __name__ == '__main__':
    app.run()