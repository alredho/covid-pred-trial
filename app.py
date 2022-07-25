from flask import Flask, render_template, url_for, request, redirect
import numpy as np
import pickle

app = Flask(__name__)

load_model = pickle.load(open('covid_xgboost_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods = ['POST'])
def test():
    data = request.form.to_dict()
    name = data['inputName']    
    fin_data = np.array([
        [data['radioCough'], data['radioFever'], data['radioSoreThroat'], data['radioShortBreath'], 
        data['radioHeadache'], data['radioAge'], data['radioGender']]
        ])
    fin_data = fin_data.astype('int')
    prediction = load_model.predict(fin_data)
    
    if prediction == 0:
        prediction = "Negative"
    elif prediction == 1:
        prediction = "Positive"

    return render_template('result.html', prediction = prediction, name = name)

if __name__ == "__main__":
    app.run(debug=True)