from flask import Flask
from database import db

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_pyfile("config.py", silent=True)
    db.init_app(app)
    return app