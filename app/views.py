import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from flask import render_template, request, flash, redirect, url_for
from wtforms.validators import ValidationError
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
            flash('Please enter all fields', 'warning')
            return render_template("index.html")
        else:
            user_premise = request.form["premise"]
            user_hypothesis = request.form["hypothesis"]
            user_model = request.form["model"]

            excluded_chars = "*^+%&/()=}][{$#"
            for char in user_premise:
                if char in excluded_chars or len(user_premise) < 3:
                    flash('Please not enter special characters in premise and at least 3 characters', 'warning')
                    return render_template("index.html")
            for char in user_hypothesis:
                if char in excluded_chars or len(user_premise) < 3:
                    flash('Please not enter special characters in hypothesis and at least 3 characters', 'warning')
                    return render_template("index.html")

            if user_model == "BERT":
                (premise_rp, hypothesis_rp), (premise_translated, hypothesis_translated), (premise_clean, hypothesis_clean), (premise, hypothesis), (index, mask, token_type), (prob, pred) = bert_inference(user_premise, user_hypothesis)
                result = NLIDatabase(premise=user_premise,
                                    hypothesis=user_hypothesis,
                                    premise_cleaned=premise_clean,
                                    hypothesis_cleaned=hypothesis_clean,
                                    probability=prob,
                                    prediction=pred)
                db.session.add(result)
                db.session.commit()
                return render_template("predict.html", user_premise=user_premise, user_hypothesis=user_hypothesis, premise_rp=premise_rp, hypothesis_rp=hypothesis_rp, premise_translated=premise_translated, hypothesis_translated=hypothesis_translated, premise_clean=premise_clean, hypothesis_clean=hypothesis_clean, premise=premise, hypothesis=hypothesis, index=index, mask=mask, token_type=token_type, prob=prob, pred=pred)
                
            elif user_model == "RoBERTa":
                (premise, hypothesis), (input_ids, attention_mask), decode, (pred, prob) = roberta_inference(user_premise, user_hypothesis)
                result = NLIDatabase(premise=user_premise,
                                    hypothesis=user_hypothesis,
                                    premise_cleaned=premise,
                                    hypothesis_cleaned=hypothesis,
                                    probability=prob,
                                    prediction=pred)
                db.session.add(result)
                db.session.commit()
                return render_template("predict_roberta.html", premise=user_premise, hypothesis=user_hypothesis,
                           premise_cl=premise, hypothesis_cl=hypothesis, decoded=decode, index=input_ids,
                           mask=attention_mask, prob=prob, pred=pred)

@app.route("/database", methods=(["GET", "POST"]))
def database():
    if request.method == 'POST':
        if request.form.get('delete') == 'Delete All Database':
            drop_table()
            db.create_all(app=create_app())
            db.session.commit()
    return render_template("database.html", results=NLIDatabase.query.all())

@app.route('/delete/<int:id>')
def delete(id):
    id_delete = NLIDatabase.query.get_or_404(id)
    db.session.delete(id_delete)
    db.session.commit()
    return redirect(url_for('database'))

@app.route("/document")
def document():
    return render_template("document.html")

@app.route("/sample")
def sample():
    # header = ['premise', 'hypothesis', 'language','label']
    df = pd.read_csv("data/samples.csv")
    premises = df['premise'].to_list()
    hypothesises = df['hypothesis'].to_list()
    languages = df['language'].to_list()
    labels = df['label'].to_list()
    return render_template("sample.html", zip=zip, premises=premises, hypothesises=hypothesises, languages=languages, labels=labels)
if __name__ == '__main__':
    create_table()
    app.run(host=app.config['HOST'], port=app.config['PORT'])