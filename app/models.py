from database import db

class NLIDatabase(db.Model):
    id = db.Column("id", db.Integer, primary_key=True)
    premise = db.Column(db.String(1000))
    hypothesis = db.Column(db.String(1000))
    prediction = db.Column(db.String(10))

    def __repr__(self):
        return f"{self.id}-{self.prediction}"