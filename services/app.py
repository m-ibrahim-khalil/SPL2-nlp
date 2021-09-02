from flask import Flask
import flask.scaffold
from flask_jwt_extended import JWTManager

flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
from flask_restful import Api
from flask_bcrypt import Bcrypt
from services.resources.routes import initialize_routes
from database.db import initialize_db
from flask_cors import CORS
from model.model import initializeModel

app = Flask(__name__)
CORS(app)

app.config["JWT_SECRET_KEY"] = 'top_secret'
api = Api(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)


app.config['MONGODB_SETTINGS'] = {
    'host': 'mongodb://localhost/question-answer'
}

initialize_db(app)
initialize_routes(api)
initializeModel()

app.run(host='127.0.0.1', port=5001, debug=False)
