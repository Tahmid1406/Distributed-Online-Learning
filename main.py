from flask import Flask
from flask import render_template, session, request, redirect, g, url_for 
from blockchain import *
from flask_sqlalchemy import SQLAlchemy 
from datetime import datetime
import pickle
import os
import numpy as np
import pandas as pd
import os
import math
from numpy import * 
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score # import accuracy metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")




app = Flask(__name__)

app.secret_key = os.urandom(24)


MODEL = pickle.load(open('Models/DSLModel_1.pkl', 'rb'))

MODEL_tweak = MODEL

USER_DIR = 'users/'
MODEL_DIR = 'static/models/'

def updatehash(*args):
    hashing_text = ""; h = sha256()

    #loop through each argument and hash
    for arg in args:
        hashing_text += str(arg)

    h.update(hashing_text.encode('utf-8'))
    return h.hexdigest()



@app.route('/query', methods=['POST', 'GET'])
def make_query():
    prediction = None

    if request.method == 'POST':
        t_type = request.form.get('type')
        amount = request.form.get('amount')
        oldBalanceOrig = request.form.get('oldBalanceOrig')
        newBalanceOrig = request.form.get('newBalanceOrig')
        oldBalancedest = request.form.get('oldBalancedest')
        newBalancedest = request.form.get('newBalancedest')
        
        data = [
            np.array([
            int(t_type), 
            float(amount), 
            float(oldBalanceOrig), 
            float(newBalanceOrig), 
            float(oldBalancedest), 
            float(newBalancedest)
            ])
            ]
        result = MODEL.predict(data)

        if result == [1]:
            prediction = "Fraud"
        else:
            prediction = "Non-Fraud"

    return render_template('makeQuery.html', pred=prediction)



@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html', user= g.user, role=g.role)


@app.route('/initialTraining', methods=['POST', 'GET'])
def initialTraining():
    fileno = len(os.listdir(MODEL_DIR))
    if request.method == 'POST':
        trainFile = request.files['trainFile']
        testFile = request.files['testFile'] 
        
        train_filepath = os.path.join(MODEL_DIR, "train.csv")
        test_filepath = os.path.join(MODEL_DIR, "test.csv")
        
        trainFile.save(train_filepath)
        testFile.save(test_filepath)
    return render_template('initialTraining.html', user= g.user, role=g.role)
    

@app.route('/dashboard', methods=['POST', 'GET'])
def dashboard():
    return render_template('dashboard.html')


@app.route('/login' , methods=['POST', 'GET'])
def login():
    login_text = None
    if request.method == 'POST':
        session.pop('user', None)
        uname = request.form.get('name')
        passw = request.form.get('password')
        users = sorted(os.listdir(USER_DIR), key=lambda x : int(x))
        
        for user in users[0:]:
            with open(USER_DIR + user) as f:
                user = json.load(f)
                if (user.get('name') == uname and user.get('pass1')== passw):
                    session['user'] = user.get('hash')
                    session['role'] = user.get('role')
                    return redirect(url_for('index'))
                else:
                    login_text = "username " + uname + " not found in the datebase"
    
    return render_template('login.html', not_found = login_text)


@app.route('/register' , methods=['POST', 'GET'])
def register():
    user_role = None
    if request.method == 'POST':
        name = request.form.get('name')
        pass1 = request.form.get('password')
        pass2 = request.form.get('password_2')
        role = request.form.get('role')
        hash = updatehash(name, pass1)
        if role == '1':
            user_role = 'Regulator'
        elif role == '2':
            user_role = 'Contributor'
        elif role == '3':
            user_role = 'User'
       
        user = {
            'name' : name,
            'pass1' : pass1,
            'pass2' : pass2,
            'hash' : hash,
            'role' : user_role
        }
        user_no = len(os.listdir(USER_DIR))

        if name:
            with open(USER_DIR +  str(user_no), 'w') as f:
                json.dump(user, f, indent=4, ensure_ascii=False)
                f.write('\n')
            
    return render_template('register.html')


@app.route('/train' , methods=['POST', 'GET'])
def train():
    if request.method == 'POST':
        file = request.files['csvfile']
        print(type(file))
        

    return render_template('train.html')


@app.route('/metrics' , methods=['POST', 'GET'])
def view_metrics():
    
    return render_template('metrics.html')



@app.route('/analysis' , methods=['POST', 'GET'])
def view_analysis():
    
    return render_template('analysis.html')



@app.route('/printchain')
def print_blockchain():
    
    files = sorted(os.listdir(BLOCKCHAIN_DIR), key=lambda x : int(x))
    
    results = []

    for file in files[0:]:
        with open(BLOCKCHAIN_DIR + file) as f:
            block = json.load(f)
        results.append(block)


    return render_template('currentChain.html', current_chain = results)

@app.before_request
def before_request():
    g.user = None
    g.role = None

    if 'user' in session:
        g.user = session['user']
        g.role = session['role']

if __name__ == '__main__':
    app.run(debug=True)