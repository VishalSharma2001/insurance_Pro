import numpy as np
from flask import Flask,render_template,request,redirect,url_for
import pickle
import math
app=Flask(__name__)
model=pickle.load(open("Model.pkl","rb"))

@app.route("/")
def home():
	return (render_template('Main.html'))
@app.route("/About")
def About():
	return(render_template('About1.html'))

@app.route('/predict',methods=["POST"])
def predict():
	if request.method=="POST":
		int_feature=[x for x in request.form.values()]
	#print(request.form.get(values))
	final_feature=[np.array(int_feature)]
	x= np.asarray(final_feature, dtype='float64')
	prediction=model.predict(x)
	
    #output=round(prediction[0],2)
	return render_template('Main.html',prediction_text="Your Insurance Price is {}".format(math.floor(prediction)))

if __name__=="__main__":
	app.run(debug=True)