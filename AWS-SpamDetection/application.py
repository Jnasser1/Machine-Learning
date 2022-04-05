from flask import Flask, request
import joblib

application = Flask(__name__)

vectorizer=joblib.load("vectorizer.pkl")
spam_detect_model = joblib.load("spam_detect_model.pkl")


@application.route('/')
def hello_world():
    return "Hey there, welcome to the world!"


@application.route('/spamdetect', methods = ['GET', 'POST'])
def spamdetect():
    message = request.args.get("message") 
    vectorized_message = vectorizer.transform([message])       # Vectorize the recieved message, to be passed to model.
    result = spam_detect_model.predict(vectorized_message)[0]  # The model returns an array object.
    return result



if __name__ == '__main__':
    application.run()