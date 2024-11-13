import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

dataset = pd.read_csv('./slr.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

lr = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y)

lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

print(y_pred)

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, lr.predict(x_train), color='blue')
plt.show()

print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("R2 Score: ", lr.score(x_test, y_test))
