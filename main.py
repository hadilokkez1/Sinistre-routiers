import pickle
import numpy as np

from flask import render_template, redirect, url_for,request
from flask import Flask
RF=pickle.load(open('C:/Users/hadil/PycharmProjects/soutenance/modell.pkl','rb'))
clf=pickle.load(open('C:/Users/hadil/PycharmProjects/soutenance/modelll.pkl','rb'))

knn=pickle.load(open('C:/Users/hadil/PycharmProjects/soutenance/model.pkl','rb'))
app = Flask(__name__)
@app.route("/", methods=['GET','POST'])
def home():
 return render_template('index.html')
@app.route("/about", methods=['GET','POST'])
def about():
 return render_template('about.html')

@app.route("/contact", methods=['GET','POST'])
def contact():
 return render_template('contact.html')

@app.route("/pourcentage", methods=['GET','POST'])
def pourcentage():
 return render_template('elements.html')
@app.route("/groupe", methods=['GET','POST'])
def pourcentagee():
 return render_template('groupe.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        a = np.zeros(6)
        a[0] = int(request.form['natureDuSinistre'])
        a[1] = int(request.form['age'])
        a[2] = int(request.form['sexe'])
        a[3] = int(request.form['typeImmatriculation'])
        a[4] = int(request.form['energie'])

        return render_template('resultsaison.html', prediction=knn.predict([a])[0])

@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        preg = int(request.form['CIN'])
        glucose = int(request.form['age'])
        bp = int(request.form['sexe'])
        st = int(request.form['jour'])
        bmi = float(request.form['annee'])
        dpf = float(request.form['saison'])


        data = np.array([[preg, glucose, bp, st, bmi, dpf]])
    return render_template('resultnature.html', predictionn=RF.predict(data))
@app.route('/predicttt', methods=['POST'])
def predicttt():
    if request.method == 'POST':
       aa = int(request.form['CIN'])
       bb = int(request.form['age'])
       cc = int(request.form['sexe'])
       dd = int(request.form['natureDuSinistre'])
       ee = int(request.form['jour'])
       ff = int(request.form['mois'])
       gg = int(request.form['annee'])
       hh = int(request.form['saison'])

    data = np.array([[aa, bb, cc, dd, ee, ff, gg, hh]])
    return render_template('resultp.html', predictionnn=clf.predict(data))

if __name__== "__main__":
    app.run(host='localhost', port=5000)

