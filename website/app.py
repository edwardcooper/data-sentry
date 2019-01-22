from flask import Flask
from flask import render_template
from flask import flash
from flask import redirect
from flask import url_for

from user_register_forms import RegistrationForm, LoginForm

from flask_sqlalchemy import SQLAlchemy


# libraries for the model part
from sklearn.externals import joblib
from piidetect.pipeline import text_clean, word_embedding
import pandas as pd

# word2vec_model = joblib.load("word2vec_pipe_cv_production.pkl")






app = Flask(__name__)
app.config['SECRET_KEY'] = '34d9f494c01b69e2f6aad1e9ccfd5bb48630ab9b'



@app.route('/')
def home_page():
    return render_template("home_page.html")

@app.route("/API")
def API_page():
    return render_template("API_page.html")

@app.route("/Login")
def Login_page():
    return render_template("Login_page.html")

@app.route('/register_form', methods = ['GET',"POST"])
def register_form():
    form_for_register = RegistrationForm()
    if form_for_register.validate_on_submit():
        flash(f"Account create for {form_for_register.username.data}!", 'success')
        return redirect(url_for("API_page"))
    return render_template('register_form.html', form = form_for_register)

@app.route('/login_form', methods = ['GET',"POST"])
def login_form():
    form_for_login = LoginForm()
    if form_for_login.validate_on_submit():
        if form_for_login.email.data == "admin@piisentry.com" and form_for_login.password.data == "1234":
            flash("You have been loged in!", 'success')
            return redirect(url_for("API_page"))
        else:
            flash("Please double check your email and password", "danger")

    return render_template('login_form.html', form = form_for_login)



if __name__ == "__main__":
    app.run(host = '0.0.0.0', debug = True, port = 80)
