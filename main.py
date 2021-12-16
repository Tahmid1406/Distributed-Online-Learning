from flask import Flask
from flask import render_template
from flask import request
from blockchain import *
from flask_sqlalchemy import SQLAlchemy 
from datetime import datetime


app = Flask(__name__)

REGISTER = False



@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        t_type = request.form.get('type')
        amount = request.form.get('amount')
        oldBalanceOrig = request.form.get('oldBalanceOrig')
        newBalanceOrig = request.form.get('newBalanceOrig')
        oldBalancedest = request.form.get('oldBalancedest')
        newBalancedest = request.form.get('newBalancedest')
        
        # we have to conduct machine learning works here

        # write_block(borrower=borrower, lender=lender, amount=amount)
    return render_template('index.html')

@app.route('/' , methods=['POST', 'GET'])
def register_user():
    if request.method == 'POST':
        name = request.form.get('name')
        pass1 = request.form.get('password')
        pass2 = request.form.get('password_2')
        print(name)    
        print(pass1)    
        print(pass2)    
    return render_template('index.html')



@app.route('/register' , methods=['POST', 'GET'])
def register():
    return render_template('register.html')



@app.route('/checking')
def print_blockchain():
    
    files = sorted(os.listdir(BLOCKCHAIN_DIR), key=lambda x : int(x))
    
    results = []

    for file in files[0:]:
        with open(BLOCKCHAIN_DIR + file) as f:
            block = json.load(f)
        results.append(block)


    return render_template('index.html', current_chain = results)

if __name__ == '__main__':
    app.run(debug=True)