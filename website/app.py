from flask import Flask
from flask import render_template
from flask import flash
from flask import redirect
from flask import url_for

from user_entry_forms import RegistrationForm, LoginForm, TextEntry

from flask_sqlalchemy import SQLAlchemy
from piidetect.pipeline import text_clean, word_embedding

# libraries for the model part
from sklearn.externals import joblib
from piidetect.pipeline import clean_text, word_embedding
import pandas as pd
# for caching the word2vec model

from werkzeug.contrib.cache import SimpleCache
cache = SimpleCache()

app = Flask(__name__)
# this a temp setup for development.
app.config['SECRET_KEY'] = '34d9f494c01b69e2f6aad1e9ccfd5bb48630ab9b'



def get_word2vec_model(filename):
    word2vec_model = cache.get("word2vec_pipe_cv_production")
    if word2vec_model is None:
        word2vec_model = joblib.load(filename)
        cache.set("word2vec_pipe_cv_production", word2vec_model, timeout = 300)

    return word2vec_model

@app.route('/', methods = ["GET", "POST"])
def home_page():
    form_for_text_input = TextEntry()
    if form_for_text_input.text.data is None:
        text = pd.Series(["My email address is edwardlu@gmail.com"])
        output = text.iloc[0]
    else:
        text = pd.Series(form_for_text_input.text.data)
        # load the model if not cached yet.
        word2vec_model = get_word2vec_model("word2vec_pipe_cv_production.pkl")
        # model prediction
        model_result = word2vec_model.predict(text)
        # convert the results into different messages. 
        if model_result[0] == 1:
            flash("The text you entered contains PII data","danger")

        elif model_result[0] == 0:
            flash("The text you entered does not contain PII data",'success')

    return render_template("home_page.html", form = form_for_text_input)

@app.route("/API")
def API_page():
    return render_template("API_page.html")


@app.route('/Register', methods = ['GET',"POST"])
def Register_page():
    form_for_register = RegistrationForm()
    if form_for_register.validate_on_submit():
        flash(f"Account create for {form_for_register.username.data}!", 'success')
        return redirect(url_for("API_page"))
    return render_template('Register_page.html', form = form_for_register)

@app.route('/Login', methods = ['GET',"POST"])
def Login_page():
    form_for_login = LoginForm()
    if form_for_login.validate_on_submit():
        if form_for_login.email.data == "admin@piisentry.com" and form_for_login.password.data == "1234":
            flash("You have been loged in!", 'success')
            return redirect(url_for("API_page"))
        else:
            flash("Please double check your email and password", "danger")

    return render_template('Login_page.html', form = form_for_login)



if __name__ == "__main__":
    app.run(host = '0.0.0.0', debug = True, port = 80)
