# ANN Regression with tensorflow on Render.com

# Demo
- call: https://linreg.onrender.com
- Note: Please wait for approx. 2 min. -> till the queued webservice (due to long inactivity) is restarted
- Then Enter: 998 for feature1 and 1000 for feature2, hit predict, then check result: 425.12085, further values to check are in fake_reg.csv!

# Jupyter Notebook is used only with python 3.7.7 intel mac legacy

# installation (optional use to regenerate model & scaler)
- download and install miniconda from website: https://docs.anaconda.com/miniconda/
- `conda create --name tensorflow377 python=3.7.7`
- `conda info --envs`
- `conda activate tensorflow377`
- `conda install ipykernel`
- `python -m ipykernel install --user --name tensorflow377 --display-name "Python 3.7.7 (tensorflow)"`

# installation local of Jupyter Notebook (optional)
- `conda deactivate`
- `conda uninstall -y jupyter`

 # (Use pip if using legacy Python 3.7.7) locally on intel mac (optional)
- `conda activate tensorflow377`
- `pip3 install --upgrade pip`
- `pip3 install jupyter`
- `pip install notebook --upgrade`
- `pip install Jinja2==3.0.3`
- `pip install MarkupSafe==2.0.0`
- `pip install zipp==3.1.0`
- `pip3 install chardet`
- `conda install -c anaconda importlib-metadata`


- `conda activate tensorflow377`
- `conda install -y pandas`
- `conda install -y seaborn`
- `conda install -y matplotlib`
- `conda install -y tensorflow`
- run notebook: `jupyter notebook`

# fork this repo on github.com
- connect to the github repo from render.com (login with github account)

# To install the app use render.com
- Choose free plan 0$ and type: webservice

# build on render.com
- `pip install --upgrade pip && pip install -r requirements.txt`

# start command set on render.com
- `python app.py`

# under environment set on render.com
- PYTHON_VERSION => 3.7.7
- PORT => 5000

# already saved - you don't need to save requirements.txt locally on intel mac (optional)
- `pip freeze > requirements.txt`

# scaler (optional)
```
import pickle
scalerfile = 'scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))
new_gem = scaler.transform(new_gem)

# use the scaler
scaler = pickle.load(open('scaler.sav', 'rb'))
new_gem2 = [[feat1,feat2]]
new_gem2 = scaler.transform(new_gem2)
predict=model.predict(new_gem2) 
```