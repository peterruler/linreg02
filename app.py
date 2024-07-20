from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField
from wtforms.validators import NumberRange
from tensorflow.keras.models import load_model  
import numpy as np 
import pickle

def return_prediction(model,scaler,sample_json):
    
    feat1 = sample_json['feat1']
    feat2 = sample_json['feat2']
    
    new_gem2 = [[feat1,feat2]]
    new_gem2 = scaler.transform(new_gem2)
    
    predict=model.predict(new_gem2) 
    
    return predict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

flower_model = load_model("my_model.h5")
scaler = pickle.load(open('scaler.sav', 'rb'))

class FlowerForm(FlaskForm):
    feat1 = StringField('Feature 1')
    feat2 = StringField('Feature 2')

    submit = SubmitField('Predict')

@app.route('/', methods=['GET', 'POST'])
def index():

    form = FlowerForm()
    if form.validate_on_submit():

        session['feat1'] = form.feat1.data
        session['feat2'] = form.feat2.data

        return redirect(url_for("prediction"))

    return render_template('home.html', form=form)


@app.route('/prediction')
def prediction():
    content = {}

    content['feat1'] = float(session['feat1'])
    content['feat2'] = float(session['feat2'])

    results = return_prediction(model=flower_model,scaler=scaler,sample_json=content)

    return render_template('prediction.html',results=results[0][0])


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000) 
