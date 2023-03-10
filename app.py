import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app=Flask(__name__)
model=pickle.load(open('random.sav','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    output = model.predict(data)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[(x) for x in request.form.values()]
    print(data)
    output = model.predict(data)
    return render_template("home.html",prediction_text="This mail is categorized as : {}".format(output))


if __name__=="__main__":
    app.run(debug=True)