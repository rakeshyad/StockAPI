
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from flask import Flask,render_template,request
import pickle

model1=pickle.load(open("Nifty-SVM-all","rb"))
model2=pickle.load(open("MSFT-SVM-all","rb"))

app = Flask(__name__)

# Home Page is good 

@app.route('/')
def home():
    return render_template("index.html")

#Nifty Page

@app.route('/Nifty',methods=['GET'])
def Nifty():
    
    return render_template('nifty.html')

#MSFT Page

@app.route('/MSFT',methods=['GET'])
def MSFT():
    
    return render_template('MSFT.html')

#Model1 Prediction  

@app.route('/predict1',methods=['POST'])
def predict1():
    
    
        day = request.form.get("day")
        month =  request.form.get("Month")
        time =request.form.get("Time")
        open =request.form.get("Open")
        data1 =day
        data2 =month
        data3 =time
        data4 =float(open)

        data=pd.DataFrame({"day":data1,"month":data2,"Time":data3,"open":data4},index=[0])
        data=np.array(data)
        prediction=model1.predict(data)
        prediction = round(prediction[0], 2)
        return render_template('nifty.html', prediction_text='Price will be  {}'.format(prediction))





#Model2 Prediction 


@app.route('/predict2',methods=["POST"])
def predict2():
    
    
        day = request.form.get("day")
        month =  request.form.get("Month")
        Year =request.form.get("Year")
        open =request.form.get("Open")
        data1 =day
        data2 =month
        data3 =Year
        data4 =float(open)

        data=pd.DataFrame({"day":data1,"month":data2,"Year":data3,"open":data4},index=[0])
        data=np.array(data)
        prediction=model2.predict(data)
        prediction = round(prediction[0], 2)
        return render_template('MSFT.html', prediction_text='Price will be  {}'.format(prediction))

#ports not working properly 

if __name__=="__main__":
    app.run(debug=True,port=5000)

