import flask
from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import pickle

with open('model.pkl','rb') as f:
    p=pickle.load(f)

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    a = [float(i) for i in request.form.values()]
    s = p.predict([a])
    return render_template('index.html', prediction=float(s[0]))
if __name__=='__main__':
    app.run(debug=True)