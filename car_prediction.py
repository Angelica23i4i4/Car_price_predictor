

import pandas as pd
import pickle

# из sklearn мы импортируем алгоритм k-ближайших соседей
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('used_cars.csv')

df = df.drop('clean_title', axis=1)

df = df.drop(columns=['engine', 'int_col', 'fuel_type', 'model', 'ext_col'], axis=1)

x = df.drop('price', axis=1)
y = df['price']

y = y.apply(lambda x: int(x.replace('$', '').replace(',','')))

from category_encoders.target_encoder import TargetEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder

encoder = TargetEncoder(cols=['brand', 'transmission'], smoothing=1)
encoder1 = OneHotEncoder()
encoder2 = OrdinalEncoder()

accident = encoder1.fit_transform(x['accident'])

x['milage'] = x['milage'].apply(lambda x: int(x.replace(' mi.', '').replace(',', '')))

x['model_year'] = x['model_year'].astype(str)
year = encoder2.fit_transform(x['model_year'])

x = x.drop('model_year', axis=1)
x.insert(loc=1,column='model_year', value=year)

x1 = x.drop('accident', axis=1)
x = pd.concat([x1, accident], axis='columns', join='outer')

x = x.drop(['accident_2', 'accident_3'], axis=1)

x = encoder.fit_transform(x,y)

df = pd.concat([x,y], axis='columns', join='outer')

scaler = StandardScaler()
df_new = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

x = df_new.drop('price', axis=1)
y = df_new['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=125)

model = LinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)

MSE = mean_squared_error(pred, y_test)

pickle.dump(model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

