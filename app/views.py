import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print(sys.path)
from flask import render_template, request, flash
from __init__ import create_app
from models import NLIDatabase
from database import db, drop_table
from bert.inference import bert_inference
from roberta.roberta_inference import roberta_inference


app = create_app()

@app.before_first_request
def create_table():
    db.create_all(app=create_app())

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test")
def test():
    return render_template("predict_roberta.html", premise='Whats up nigga', hypothesis='hello world',
                           premise_cl='hello world', hypothesis_cl='hello world', index='hello world',
                           mask='hello world', prob='hello world', pred='hello world')

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
            elif user_model == "RoBERTa":
                (premise, hypothesis), (input_ids, attention_mask), decode, (pred, prob) = roberta_inference(user_premise, user_hypothesis)
                print(f"Prob {prob}")
                print(f"Pred {pred}")
                
                return render_template("predict_roberta.html", premise=user_premise, hypothesis=user_hypothesis,
                           premise_cl=premise, hypothesis_cl=hypothesis, decoded=decode, index=input_ids,
                           mask=attention_mask, prob=prob, pred=pred)
                
if __name__ == '__main__':
    app.run()