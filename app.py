from json import load
from flask import Flask, render_template, url_for, request, redirect
import numpy as np
import pickle
from model import Xgboost

app = Flask(__name__)

load_model = pickle.load(open('covid_xgboost_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classifier', methods = ['POST'])
def classifier():
    data = request.form.to_dict()
    name = data['inputName']    
    fin_data = np.array([
        [data['radioCough'], data['radioFever'], data['radioSoreThroat'], data['radioShortBreath'], 
        data['radioHeadache'], data['radioAge'], data['radioGender']]
        ])
    fin_data = fin_data.astype('int')
    
    xgboost_model = Xgboost(load_model)
    result = xgboost_model.classify(fin_data)

    return render_template('result.html', classification = result, name = name)

if __name__ == "__main__":
    app.run(debug=True)