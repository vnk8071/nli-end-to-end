import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import render_template, request, flash, redirect, url_for
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
                (premise_rp, hypothesis_rp), (premise_translated, hypothesis_translated), (premise_clean, hypothesis_clean), (premise, hypothesis), (index, mask, token_type), (prob, pred) = bert_inference(user_premise, user_hypothesis)
                result = NLIDatabase(premise=user_premise,
                                    hypothesis=user_hypothesis,
                                    probability=prob,
                                    prediction=pred)
                db.session.add(result)
                db.session.commit()
                flash('Record was successfully added')
                return render_template("predict.html", user_premise=user_premise, user_hypothesis=user_hypothesis, premise_rp=premise_rp, hypothesis_rp=hypothesis_rp, premise_translated=premise_translated, hypothesis_translated=hypothesis_translated, premise_clean=premise_clean, hypothesis_clean=hypothesis_clean, premise=premise, hypothesis=hypothesis, index=index, mask=mask, token_type=token_type, prob=prob, pred=pred)


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

if __name__ == '__main__':
    create_table()
    app.run()