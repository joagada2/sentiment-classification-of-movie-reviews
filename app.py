from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import joblib
import nltk
import re
import numpy as np

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

model = joblib.load('agada_sentiment_classifier')
vectorizer = joblib.load('agada_sentiment_vectorizer')

@app.route('/', methods=['POST'])
def my_form_post():
    text1 = request.form['text1'].lower()
    #text1 = str(text1)
    #text1 = re.sub(r"@\S+ ", r' ',text1)
    #text1 = re.sub('https://.*','',text1)
    #text1 = re.sub("\s+",' ',text1)
    #text1 = re.sub("\n+",' ',text1)
    #text1 = re.sub("[^a-zA-Z]",' ',text1)
    #text1 = text1.split(' ')
    #text1 = (vectorizer.transform(text1))
    #compound = model.predict(text1)
    compound = model.predict(vectorizer.transform(np.array([text1])))[0]
    return compound #compound

    return render_template('form.html', final=compound, text1=text1)

if __name__ == "__main__":
    app.run(debug=True)