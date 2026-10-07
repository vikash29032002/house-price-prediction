import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'SquareFeet': [1500, 1800, 2400, 3000, 3500, 1200, 2000, 2800],
    'Bedrooms': [3, 3, 4, 4, 5, 2, 3, 4],
    'Age': [10, 5, 2, 1, 8, 15, 7, 4],
    'Price': [300000, 360000, 480000, 590000, 640000, 240000, 410000, 540000]
}

df = pd.DataFrame(data)

X = df[['SquareFeet', 'Bedrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")