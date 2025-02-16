# ANN Regression with TensorFlow on Render.com

## Demo
- URL: [https://linreg.onrender.com](https://linreg.onrender.com)
- Note: Please wait about 2 minutes for the web service to restart if it has been inactive for a long time.
- Then enter: 998 for feature1 and 1000 for feature2, click on "predict" and check the result: 425.12085. More values to check can be found in the `fake_reg.csv` file.

## Jupyter Notebook
- Using Python 3.7.7 on an Intel Mac.
- Eventually just works on Intel PC on CPU, please try

## Installation (optional, to regenerate model and scaler)
1. Download and install Miniconda from the website: [Miniconda Documentation](https://docs.anaconda.com/miniconda/)
2. Create a new environment:
    ```sh
    conda create --name tensorflow377 python=3.7.7
    conda info --envs
    conda activate tensorflow377
    conda install ipykernel
    python -m ipykernel install --user --name tensorflow377 --display-name "Python 3.7.7 (tensorflow)"
    ```
3. Load the pip dependencies in Order to retrain model and recreate scaler:

    ```pip install -r requirements-jpynb.txt```
## Local Installation of Jupyter Notebook (optional)
1. Deactivate the environment:
    ```sh
    conda deactivate
    ```
2. Uninstall Jupyter:
    ```sh
    conda uninstall -y jupyter
    ```
3. Install Jupyter and other dependencies:
    ```sh
    conda activate tensorflow377
    pip3 install --upgrade pip
    pip3 install jupyter
    pip install notebook --upgrade
    pip install Jinja2==3.0.3
    pip install MarkupSafe==2.0.0
    pip install zipp==3.1.0
    pip3 install chardet
    conda install -c anaconda importlib-metadata
    conda install -y pandas
    conda install -y seaborn
    conda install -y matplotlib
    conda install -y tensorflow
    ```

4. Start the notebook:
    ```sh
    jupyter notebook
    ```

## Deploying the Application on Render.com
1. Fork this repository on GitHub.
2. Connect the GitHub repository to Render.com (sign in with your GitHub account).
3. Choose the free plan (0$) and the type "Web Service".
4. Build the application on Render.com:
    ```sh
    pip install --upgrade pip && pip install -r requirements.txt
    ```

## Set Start Command on Render.com
- `python app.py`

## Set Environment Variables on Render.com
- `PYTHON_VERSION` => `3.7.7`
- `PORT` => `5000`

## Save Requirements (optional)
- `pip freeze > requirements.txt`

## Scaler (optional)
```python
import pickle

# Save scaler
scalerfile = 'scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))

# Load scaler
scaler = pickle.load(open('scaler.sav', 'rb'))
new_gem2 = [[feat1, feat2]]
new_gem2 = scaler.transform(new_gem2)
predict = model.predict(new_gem2)