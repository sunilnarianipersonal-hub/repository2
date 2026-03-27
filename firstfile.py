import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
y_pred = lr.predict(x_test)
meanSquaredError = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {meanSquaredError}') 
print(f'R^2 Score: {r2}')


# print(df.head())