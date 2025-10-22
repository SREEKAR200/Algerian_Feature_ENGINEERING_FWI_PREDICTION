from flask import Flask,request,render_template,jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
application=Flask(__name__)
app=application

ridge_model=pickle.load(open('ridge.pkl','rb'))
standard_scalar=pickle.load(open('scaler.pkl','rb'))
@app.route("/")
def index():
    return render_template('index.html')
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        mon = float(request.form.get('mon'))
        temp = float(request.form.get('temp'))
        rh = float(request.form.get('rh'))
        ws = float(request.form.get('ws'))
        rain = float(request.form.get('rain'))
        ffmc = float(request.form.get('ffmc'))
        dmc = float(request.form.get('dmc'))
        isi = float(request.form.get('isi'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Apply standard scaling
        scaled_data = standard_scalar.transform([[mon,temp,rh,ws,rain,ffmc,dmc,isi,Classes,Region]])
        prediction = ridge_model.predict(scaled_data)
        return render_template('home.html',results=prediction)
    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)