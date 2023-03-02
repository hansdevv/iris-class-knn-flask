from flask import Flask, render_template, url_for, jsonify, request
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

modelKNN = pickle.load(open('models/modelKNN.pkl','rb',))
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')

@app.route('/')
def dashboard():
	sepal_length = 4.6
	sepal_width = 3.6
	petal_length = 1
	petal_width = 0.2
	res = modelKNN.predict([[sepal_length, sepal_width, petal_length, petal_width]])
	return render_template('index.html', Predict = res[0])

@app.route('/class-iris-knn')
def index_iris_KNN():
	return render_template('indexIrisKnn.html')


@app.route('/predictKNN',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]  
    prediction = modelKNN.predict(features) 
    result = prediction[0]

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
	app.run(
		debug=True,
		host='localhost',
		port=5555
	)
