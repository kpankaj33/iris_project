from flask import Flask,jsonify
import pickle

app = Flask(__name__)


@app.route('/')
def index():
    return "deafult API"

@app.route('/predict')
def iris_pred():
     
    with open("model.pkl","rb") as model:
        ml_model = pickle.load(model)
    
    SepalLengthCm = 6 
    SepalWidthCm = 4.8
    PetalLengthCm = 5.2
    PetalWidthCm = 3.75


    result = ml_model.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
    if result[0] == 2:
        iris_flower = "Iris-virginica"

    return iris_flower


if __name__ == "__main__":
    app.run(debug=True)
