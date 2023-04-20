import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import sqlite3
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
from math import sqrt
from numpy import log
from pandas import Series
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import seaborn as sns
from datetime import datetime
import subprocess

ds_gold = 'Volume'
ds_etf = 'Close'
date_format = '%Y-%m-%d'
df = pd.read_csv("data/data_ETF.csv")

# Define exploratory variables
# Finding moving average of past 3 days and 9 days
df['S_1'] = df[ds_gold].shift(1).rolling(window=3).mean()
df['S_2'] = df[ds_gold].shift(1).rolling(window=12).mean()
df = df.dropna()
X = df[['S_1', 'S_2']]
y = df[ds_gold]
# Split into train and test
t = 0.2

# Performing linear regression
linear = LinearRegression().fit(X, y)

print("Gold Price =", round(linear.coef_[0], 2), "* 2 Month Moving Average", round(
    linear.coef_[1], 2), "* 1 Month Moving Average +", round(linear.intercept_, 2))

# Predict prices
predicted_price = linear.predict(X)

predicted_price = pd.DataFrame(
    predicted_price, index=y.index, columns=['price'])
predicted_price.plot(figsize=(10, 5))
y.plot()
plt.legend(['predicted_price', 'actual_price'])
plt.ylabel("Gold Price")
plt.savefig("static/final.png")

#plt.show()



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")

@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features= [float(x) for x in request.form.values()]
    #print(int_features,len(int_features))
    final4=[np.array(int_features)]

    #final_features = np.array([val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12,val13,val14,val15,val16,val17,val18]).reshape(1,-1)
    model = joblib.load("model.pkl")
    predict = model.predict(final4)
    predict = predict[0]
    vol = predict * 0.005540 / 3
    usd = vol / 38.6
    inr = usd * 81.74
    return render_template('result.html', output=round(inr))


@app.route('/notebook')
def notebook():
	return render_template('notebook.html')

@app.route('/about')
def about():
	return render_template('about.html')

if __name__ == "__main__":
    app.run()
