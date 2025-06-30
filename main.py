import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from car_prediction import df_new, MSE

import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
df = pd.read_csv('used_cars.csv')
df['milage'] = df['milage'].apply(lambda x: int(x.replace(' mi.', '').replace(',','')))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    brand = request.form.get('Brand')
    year = request.form.get('Year')
    milage = request.form.get('Milage')
    milage = milage.replace(' mi.', '').replace(',', '')
    accident = request.form.get('Accident')
    #transmode = df['transmission'].mode().values.tolist()
    #trans = transmode[0]

    df2 = df.join(df_new, rsuffix='_right', how='outer').reset_index(drop=True)
    brand_new = df2.loc[df2['brand'] == brand, 'brand_right']
    brand = brand_new.values.tolist()[0]
    year_new = df2.loc[df2['model_year'] == int(year), 'model_year_right']
    year = year_new.values.tolist()[0]
    milage_new = df2.loc[df2['milage'] == int(milage), 'milage_right']
    milage = milage_new.values.tolist()[0]
    #trans_new = df2.loc[df2['transmission'] == trans, 'transmission_right']
    #trans = random.choice(trans_new.values.tolist())
    if accident == 'Yes':
        accident = 'At least 1 accident or damage reported'
    else:
        accident = 'None reported'
    accident_new = df2.loc[df2['accident'] == accident, 'accident_1']
    accident = accident_new.values.tolist()[0]
    trans_new = df2.loc[(df2['accident_1'] == accident)  & (df2['model_year_right'] == year), 'transmission_right']
    trans = trans_new.values.tolist()[0]
    code_features = [brand, year, milage, trans, accident]


    final_features = [np.asarray(code_features, dtype='object')]
    prediction = model.predict(final_features)
    output = prediction[0]
    df2['mistake'] = df2['price_right'].apply(lambda x: (x-output)**2)
    price_frame = df2.loc[df2['mistake'] <= MSE, 'price']
    output = price_frame.values.tolist()[0]



    return render_template('index.html', prediction_text='Price should be $ {}'.format(output))


@app.route('/predict_api', methods= ['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)

