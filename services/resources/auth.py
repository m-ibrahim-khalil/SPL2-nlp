from flask import Response, request
from flask_jwt_extended import create_access_token
from database.models import User
from flask_restful import Resource
import datetime
from mongoengine.errors import FieldDoesNotExist, NotUniqueError, DoesNotExist
from services.resources.errors import SchemaValidationError, EmailAlreadyExistsError, UnauthorizedError, \
    InternalServerError


class SignupApi(Resource):
    def post(self):
        try:
            body = request.get_json()
            user = User(**body)
            user.hash_password()
            user.save()
            id = user.id
            return True, 200
        except FieldDoesNotExist:
            raise SchemaValidationError
        except NotUniqueError:
            raise EmailAlreadyExistsError
        except Exception as e:
            raise InternalServerError


class LoginApi(Resource):
    def post(self):
        try:
            body = request.get_json()
            username = body.get('username')
            role = body.get('role')
            password = body.get('password')
            authorized = False
            user = None
            for user in User.objects(role=role):
                if user.username == username:
                    authorized = user.check_password(password)
                    user = user
                    break

            if not authorized:
                return "", 401

            expires = datetime.timedelta(days=7)
            access_token = create_access_token(identity=str(user.id), expires_delta=expires)
            return f"Bearer {access_token}", 200
        except (UnauthorizedError, DoesNotExist):
            raise UnauthorizedError
        except Exception as e:
            raise InternalServerError
