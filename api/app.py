from flask import Flask, render_template, url_for, jsonify, request
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)

modelKNN = pickle.load(open('models/modelKNN.pkl','rb',))
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/irisKNN', methods=['GET','POST'])
def index_iris_KNN():
	if(request.method == 'POST'):
		res = ''
		sepal_length = float(request.form['sepal_length'])
		sepal_width = float(request.form['sepal_width'])
		petal_length = float(request.form['petal_length'])
		petal_width = float(request.form['petal_width'])
		pred = modelKNN.predict([[sepal_length, sepal_width, petal_length, petal_width]])
		res = pred[0]
		
		# return json.dumps(pred,cls=NumpyEncoder)
		return res
	return render_template('class_iris_KNN.html')

if __name__ == '__main__':
	app.run(
		debug=True,
		host='localhost',
		port=5555
	)
