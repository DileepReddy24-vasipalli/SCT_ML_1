import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('household.csv')


df = df.dropna()
print(df.isnull().sum())
X = df[['Area_SqFt', 'Rooms']] 
y = df['Price'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
print("Model training completed!")


joblib.dump(model, 'house_price_model.pkl')
print("Model saved successfully as 'house_price_model.pkl'!")