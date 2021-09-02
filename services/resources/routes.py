from .auth import SignupApi, LoginApi
from .context_question import ContextQuestionsApi, ContextQuestionApi, AnswersApi


# from services.reset_password import ForgotPassword, ResetPassword

def initialize_routes(api):

    api.add_resource(SignupApi, '/api/register')
    api.add_resource(LoginApi, '/api/login')

    api.add_resource(ContextQuestionsApi, '/api/cq')
    api.add_resource(ContextQuestionApi, '/api/cq/<id>')
    api.add_resource(AnswersApi, '/api/getCorrectAnswers')
