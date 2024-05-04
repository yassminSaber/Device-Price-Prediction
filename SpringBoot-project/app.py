from flask import Flask,render_template,request
from flask_bootstrap import Bootstrap 
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import MinMaxScaler
import joblib

import numpy as np

app = Flask(__name__)
Bootstrap(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///devices.db'
db = SQLAlchemy(app)


model = joblib.load('model\Device_Price_classifier.pkl')
scaler = MinMaxScaler()


@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
         battery_power = request.form["Battery Power"]
         blue = request.form["Bluetooth"]
         clock_speed = request.form["Clock Speed"]
         dual_sim = request.form["Dual SIM"]
         fc_mp = request.form["Front Camera megapixels"]
         four_g = request.form["4G"]
         internal_memory = request.form["Internal Memory"]
         mobile_depth = request.form["Mobile Depth"]
         mobile_weight = request.form["Weight of mobile phone"]
         n_cores = request.form["Number of cores"]
         pc_mp = request.form["Primary Camera megapixels"]
         pixel_resolution_h = request.form["Pixel Resolution Height"]
         pixel_resolution_w = request.form["Pixel Resolution Width"]
         ram = request.form["RAM"]
         screen_hight = request.form["Screen Height"]
         screen_width = request.form["Screen Width"]
         talk_time = request.form["talk_time"]
         three_g = request.form["3G"]
         touch_screen = request.form["Touch Screen"]
         wifi = request.form["WIFI"]
         
         feature_vector = [battery_power,blue,clock_speed,dual_sim,fc_mp,four_g,internal_memory,mobile_depth,mobile_weight,n_cores,pc_mp,pixel_resolution_h,pixel_resolution_w,ram,screen_hight,screen_width,talk_time,three_g,touch_screen,wifi]
         feature_vector = np.reshape(feature_vector,(1,20))
         feature_vector_nor = scaler.fit_transform(feature_vector)
         
         
         prediction = model.predict(feature_vector_nor)
         
         
    return render_template('results.html', prediction = prediction)             
	

if __name__ == '__main__':
	app.run(debug=True)