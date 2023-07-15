from flask import Flask,request,render_template

from src.pipeline.predict_pipeline import PredictPipeline,CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age=request.form.get('age'),
            sex=request.form.get('sex'),
            bmi=request.form.get('bmi'),
            children=request.form.get('children'),
            smoker=request.form.get('smoker'),
            region=request.form.get('region'),
        )
        pred_df = data.get_data_as_dataframe()
        pred_pipeline = PredictPipeline()
        charges = pred_pipeline.predict(pred_df)
        return render_template('home.html',charges=round(charges[0],2))
    
if __name__=='__main__':
    app.run(debug=True)