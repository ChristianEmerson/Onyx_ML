from flask import Flask,render_template,request
import pickle
import numpy as np
# from sklearn.preprocessing import LabelEncoder

app=Flask(__name__)
model=pickle.load(open('burn.pkl','rb'))
Diet=pickle.load(open('Diet.pkl','rb'))
Trainer=pickle.load(open('Trainer.pkl','rb'))

app.debug = True

@app.route('/',methods=['GET'])
def hello():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features = [float(request.form['Gender']),
                    float(request.form['Age']),
                    float(request.form['Height']),
                    float(request.form['Weight']),
                    float(request.form['Duration']),
                    float(request.form['Heart_Rate']),
                    float(request.form['Body_Temp'])]

    print("Received input values:", int_features)  
    # int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict(final)

    return render_template('index.html', prediction_text=prediction)
    # return('index.html')

@app.route('/diet',methods=['POST','GET'])
def diet():
    if request.method == 'POST':
        
        int_features = [float(request.form['age']),
                    float(request.form['weight']),
                    float(request.form['height']),
                    float(request.form['gender']),
                    float(request.form['bmi']),
                    float(request.form['bmr']),
                    float(request.form['activity_level'])]

        print("Received input values:", int_features)  
        final=[np.array(int_features)]
        dot=Diet.predict(final)
        dot=dot.astype(int)

        return render_template('Diet.html', text=dot)
    return render_template('Diet.html')

@app.route('/trainer',methods=['POST','GET'])
def trainer():
    if request.method == 'POST':
        
        int_features = [int(request.form['category']),
                    int(request.form['sleep']),
                    int(request.form['recovery']),
                    int(request.form['body']),]

        print("Received input values:", int_features)  
        final=[np.array(int_features)]
        dot=Trainer.predict(final)
        dot=dot.astype(int)

        return render_template('Trainer.html', text=dot)
    return render_template('Trainer.html')


    

                            