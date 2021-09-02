from flask import Response, request
from database.models import ContextQuestion, User
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_restful import Resource
from mongoengine.errors import FieldDoesNotExist, ValidationError, InvalidQueryError
from services.resources.errors import SchemaValidationError, InternalServerError
from model.predict_answer import PredictAnswer


class ContextQuestionsApi(Resource):

    def get(self):
        context_query = ContextQuestion.objects().to_json()
        return Response(context_query, mimetype="application/json", status=200)

    @jwt_required()
    def post(self):
        try:
            user_id = get_jwt_identity()
            body = request.get_json()
            user = User.objects.get(id=user_id)
            if user.role != 'admin':
                return "you have no permission", 401
            context_query = ContextQuestion(**body)
            context_query.save()
            return True, 200
        except (FieldDoesNotExist, ValidationError):
            raise SchemaValidationError
        except Exception as e:
            raise InternalServerError


class ContextQuestionApi(Resource):
    @jwt_required()
    def put(self, id):
        try:
            cq = ContextQuestion.objects.get(id=id)
            body = request.get_json()
            cq.update(**body)
            return {'status': 'success'}, 200
        except InvalidQueryError:
            raise SchemaValidationError
        except Exception:
            raise InternalServerError

    @jwt_required()
    def delete(self, id):
        try:
            movie = ContextQuestion.objects.get(id=id)
            movie.delete()
            return {'status': 'success'}, 200
        except Exception:
            raise InternalServerError


class AnswersApi(Resource):
    @jwt_required()
    def get(self):
        answers = []
        context_query = ContextQuestion.objects()
        # context_query = cq[len(cq) - 1]
        for cq in context_query:
            ans = []
            context = cq.context
            questions = cq.questions
            pa = PredictAnswer(context)
            for q in questions:
                ans.append(pa.predict_ans(q))
            answers.append(ans)
        print(answers)
        return answers, 200
