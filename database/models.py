from .db import db
from flask_bcrypt import generate_password_hash, check_password_hash

class User(db.Document):
    username = db.StringField(required=True, unique=True)
    email = db.EmailField(required=True, unique=True)
    password = db.StringField(required=True, min_length=6)
    role = db.StringField(required=True)

    def hash_password(self):
        self.password = generate_password_hash(self.password).decode('utf8')

    def check_password(self, password):
        return check_password_hash(self.password, password)


class ContextQuestion(db.Document):
    context = db.StringField(required=True)
    questions = db.ListField(db.StringField(), required=True)
