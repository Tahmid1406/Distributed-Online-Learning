from flask import Flask
from flask import render_template, session, request, redirect, g, url_for 
from blockchain import *
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
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score,fbeta_score # import accuracy metrics
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import precision_recall_fscore_support as score
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import io


app = Flask(__name__)

app.secret_key = os.urandom(24)

USER_DIR = 'users/'
MODEL_DIR = 'model/'
DATA_DIR = 'data/'
INC_DATA_DIR = 'incremental/'
METRIC_DIR = 'metric/'
BEST_METRIC_DIR = 'bestMetric/'


model_no = len(os.listdir(METRIC_DIR))


# Global values for metric
BEST_PRECISION = 0
BEST_RECALL = 0
BEST_FSCORE = 0
BEST_FBETA = 0
BEST_FNR = 0
CURRENT_BEST_MODEL = "model1.pkl"
CURRENT_MODEL_HASH = None
CURRENT_BLOCK_NUMBER = 0
QUERY_SERVED = 0


BLOCKCHAIN = Blockchain()

def updatehash(*args):
    hashing_text = ""; h = sha256()

    #loop through each argument and hash
    for arg in args:
        hashing_text += str(arg)

    h.update(hashing_text.encode('utf-8'))
    return h.hexdigest()

def calculateIncentive():
    pass



@app.route('/query', methods=['POST', 'GET'])
def make_query():

    model_no = len(os.listdir(MODEL_DIR))
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
        
        CURRENT_MODEL = pickle.load(open(MODEL_DIR + 'model' + str(model_no) + '.pkl', 'rb'))

        result = CURRENT_MODEL.predict(data)

        if result == [1]:
            prediction = "Fraud"
        else:
            prediction = "Non-Fraud"

        global QUERY_SERVED
        global BEST_RECALL
        global BEST_FBETA
        global BEST_FNR
        QUERY_SERVED += 1

    return render_template('makeQuery.html', pred=prediction, br=BEST_RECALL, bfb=BEST_FBETA, bfnr=BEST_FNR)



@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html', user= g.user, role=g.role)


@app.route('/initialTraining', methods=['POST', 'GET'])
def initialTraining():

    model_no = len(os.listdir(MODEL_DIR))
    text = 'True'
    
    if request.method == 'POST':
        if model_no == 0:
            text = 'True'
            trainFile = request.files['trainFile']
            testFile = request.files['testFile'] 
            
            train_filepath = os.path.join(DATA_DIR, "train.csv")
            test_filepath = os.path.join(DATA_DIR, "test.csv")
            
            trainFile.save(train_filepath)
            testFile.save(test_filepath)
            
            train = pd.read_csv('data/train.csv')
            test = pd.read_csv('./data/test.csv')

            X_train = train.drop('isFraud', axis=1)
            y_train = train.isFraud

            X_test = test.drop('isFraud', axis=1)
            y_test = test.isFraud

            X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train) 

            X_test_resampled, y_test_resampled = SMOTE().fit_resample(X_test, y_test) 

            perceptron_model_resampled = SGDClassifier(loss="perceptron", eta0=0.00001, learning_rate="constant", penalty=None)

            perceptron_model_resampled.fit(X_train_resampled, y_train_resampled)

            perceptron_train_preds = perceptron_model_resampled.predict(X_train_resampled)
            perceptron_test_preds = perceptron_model_resampled.predict(X_test_resampled)

            train_accuracy = roc_auc_score(y_train_resampled, perceptron_train_preds)
            
            test_accuracy = roc_auc_score(y_test_resampled, perceptron_test_preds)
            


            precision,recall,fscore,support = score(y_test_resampled,perceptron_test_preds,average='macro')
            fbeta = fbeta_score(y_test_resampled,perceptron_test_preds, beta=5)

            cf_matrix = confusion_matrix(y_test_resampled, perceptron_test_preds)
            CM = cf_matrix
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]

            tpr = TP/(TP+FN)
            tnr = TN/(TN+FP) 
            fpr = FP/(FP+TN)
            fnr = FN/(TP+FN)

            acc = (TP+TN)/(TP+FP+FN+TN)

            hash = updatehash('model' + str(model_no + 1) + '.pkl', precision, recall, fbeta, fnr)

            data = {
                'taining_accuracy' : train_accuracy,
                'testing_accuracy': test_accuracy,
                'overall_accuracy' : acc,
                'precision': precision,
                'recall' : recall, 
                'f1score': fscore,
                'fbeta': fbeta,
                'true_positive_rate': tpr, 
                'true_negative_rate': tnr,  
                'false_positive_rate': fpr,
                'false_negative_rate' : fnr, 
                'hash': hash
            }


            # setting up the best metrics for initial model
           

            global BEST_RECALL
            BEST_RECALL = recall

            global BEST_FBETA
            BEST_FBETA = fbeta

            global BEST_FNR
            BEST_FNR = fnr
            global CURRENT_MODEL_HASH
            CURRENT_MODEL_HASH = hash

            global BLOCKCHAIN
            global CURRENT_BLOCK_NUMBER
            CURRENT_BLOCK_NUMBER = CURRENT_BLOCK_NUMBER + 1
            

            BLOCKCHAIN.mine(Block(data, CURRENT_BLOCK_NUMBER))

            pickle.dump(perceptron_model_resampled, open(MODEL_DIR + 'model' + str(model_no + 1) + '.pkl', 'wb'))

            with open(METRIC_DIR +  str(model_no + 1), 'w') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                f.write('\n')


        else:
            text = "False"

    return render_template('initialTraining.html', user= g.user, role=g.role, training_text = text)


@app.route('/train' , methods=['POST', 'GET'])
def train():

    model_no = len(os.listdir(MODEL_DIR))

    
    if request.method == 'POST':

        file = request.files['csvfile']

        file_no = len(os.listdir(INC_DATA_DIR)) 

        filepath = os.path.join(INC_DATA_DIR + str(file_no + 1) + ".csv")
        
        file.save(filepath)

        workfile = pd.read_csv('incremental/' + str(file_no+1) + '.csv')

        train, test = train_test_split(workfile, test_size=0.33, random_state=42)

        X_train = train.drop('isFraud', axis=1)
        y_train = train.isFraud   

        X_test = test.drop('isFraud', axis=1)
        y_test = test.isFraud

        X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train) 

        X_test_resampled, y_test_resampled = SMOTE().fit_resample(X_test, y_test) 

        perceptron_model_resampled = pickle.load(open(MODEL_DIR + 'model' + str(model_no) + '.pkl', 'rb'))

        perceptron_model_resampled.partial_fit(X_train_resampled, y_train_resampled)

        perceptron_train_preds = perceptron_model_resampled.predict(X_train_resampled)
        perceptron_test_preds = perceptron_model_resampled.predict(X_test_resampled)

        train_accuracy = roc_auc_score(y_train_resampled, perceptron_train_preds)
        
        test_accuracy = roc_auc_score(y_test_resampled, perceptron_test_preds)
        


        precision,recall,fscore,support = score(y_test_resampled,perceptron_test_preds,average='macro')
        fbeta = fbeta_score(y_test_resampled,perceptron_test_preds, beta=5)

        cf_matrix = confusion_matrix(y_test_resampled, perceptron_test_preds)
        CM = cf_matrix
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]

        tpr = TP/(TP+FN)
        tnr = TN/(TN+FP) 
        fpr = FP/(FP+TN)
        fnr = FN/(TP+FN)

        acc = (TP+TN)/(TP+FP+FN+TN)

        hash = updatehash('model' + str(model_no + 1) + '.pkl', precision, recall, fbeta, fnr)

        data = {
            'taining_accuracy' : train_accuracy,
            'testing_accuracy': test_accuracy,
            'overall_accuracy' : acc,
            'precision': precision,
            'recall' : recall, 
            'f1score': fscore,
            'fbeta': fbeta,
            'true_positive_rate': tpr, 
            'true_negative_rate': tnr,  
            'false_positive_rate': fpr,
            'false_negative_rate' : fnr, 
            'hash' : hash
        }

        with open(METRIC_DIR +  str(model_no + 1), 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write('\n')

        global BEST_RECALL
        global BEST_FBETA
        global BEST_FNR
        global CURRENT_MODEL_HASH
        global BLOCKCHAIN
        global CURRENT_BLOCK_NUMBER
        

        if recall >= BEST_RECALL or fbeta >= BEST_FBETA or fnr <= BEST_FNR:
            pickle.dump(perceptron_model_resampled, open(MODEL_DIR + 'model' + str(model_no + 1) + '.pkl', 'wb'))
            CURRENT_BLOCK_NUMBER = CURRENT_BLOCK_NUMBER + 1
            CURRENT_MODEL_HASH = hash

            BLOCKCHAIN.mine(Block(data, CURRENT_BLOCK_NUMBER))


            incentive = calculateIncentive()

            if recall >= BEST_RECALL:
                BEST_RECALL = recall
            elif fbeta >= BEST_FBETA:
                BEST_FBETA = fbeta 
            elif fnr <= BEST_FNR:
                BEST_FNR = fnr
        

       
    
                
    return render_template('train.html', user= g.user, role=g.role)



@app.route('/dashboard', methods=['POST', 'GET'])
def dashboard():

    total_contribution = len(os.listdir(METRIC_DIR))
    total_model = len(os.listdir(MODEL_DIR))
    global QUERY_SERVED
    qs = QUERY_SERVED

    files = sorted(os.listdir(BLOCKCHAIN_DIR), key=lambda x : int(x))
    
    results = []

    for file in files[0:]:
        with open(BLOCKCHAIN_DIR + file) as f:
            block = json.load(f)
        results.append(block)

    met_files = sorted(os.listdir(METRIC_DIR), key=lambda x : int(x))
    
    met_results = []

    for file in met_files[0:]:
        with open(METRIC_DIR + file) as f:
            metric = json.load(f)
        met_results.append(metric)

    return render_template('dashboard.html', tot_con = total_contribution, 
            tot_model=total_model, query=qs, 
            model_hash = CURRENT_MODEL_HASH, current_chain=results, metric_list=met_results)


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





@app.route('/metrics' , methods=['POST', 'GET'])
def view_metrics():

    files = sorted(os.listdir(METRIC_DIR), key=lambda x : int(x))
    
    results = []

    for file in files[0:]:
        with open(METRIC_DIR + file) as f:
            metric = json.load(f)
        results.append(metric)

    return render_template('metrics.html', metric_list = results)


@app.route('/analysis' , methods=['POST', 'GET'])
def view_analysis(): 

    files = sorted(os.listdir(BLOCKCHAIN_DIR), key=lambda x : int(x))
    counter = 1
    counter_array = []
    acc_array = []
    recall_array = []
    fbeta_array = []
    fnr_array = []

    for file in files[0:]:
        with open(BLOCKCHAIN_DIR + str(counter)) as f:
            file = json.load(f)
            block_no = file.get('block_no')
            test_acc = file.get('data')['testing_accuracy']
            rec = file.get('data')['recall']
            fbet = file.get('data')['fbeta']
            fnr = file.get('data')['false_negative_rate']

            counter_array.append(counter)
            counter += 1
            acc_array.append(test_acc)
            recall_array.append(rec)
            fbeta_array.append(fbet)
            fnr_array.append(fnr)

    fig, axs = plt.subplots(2,1,figsize=(16,9), gridspec_kw={'height_ratios': [1, 2]})
    plt.subplot(2,2,1)
    graph = sns.lineplot(x=counter_array, y=acc_array, label="Test Accuracy Over Increments",marker="o")
    graph.set_xlabel("ML Model Increment Number", fontsize = 16)
    graph.set_ylabel("Test Accuracy Increment", fontsize = 16)

    plt.subplot(2,2,2)
    graph = sns.lineplot(x=counter_array, y=recall_array, label="Recall Over Increments",marker="s")
    graph.set_xlabel("ML Model Increment Number", fontsize = 16)
    graph.set_ylabel("Recall Update", fontsize = 16)
   

    plt.subplot(2,2,3)
    graph = sns.lineplot(x=counter_array, y=fbeta_array, label="F-Beta over Increments", marker="X")
    graph.set_xlabel("ML Model Increment Number", fontsize = 16)
    graph.set_ylabel("F-Beta Update", fontsize = 16)
   

    plt.subplot(2,2,4)
    graph = sns.lineplot(x=counter_array, y=fnr_array, label="FNR over Increments", marker="o")
    graph.set_xlabel("ML Model Increment Number", fontsize = 16)
    graph.set_ylabel("FNR Update", fontsize = 16)
    plt.savefig('1.png', bbox_inches="tight")

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