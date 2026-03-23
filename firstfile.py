import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('house_prices.csv')
df = df.drop(['id'], axis=1)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df = df.drop(['date'], axis=1)
# print(df.isnull().sum())
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['house_age'] = df['year'] - df['yr_built']
df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)
df = df.drop(['yr_built', 'yr_renovated'], axis=1)

x = df.drop(['price'], axis=1)
y = df['price'] 
x = pd.get_dummies(x, drop_first=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(x_train, y_train)




print(df.head())