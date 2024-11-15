import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_csv('./poly1.csv')
print(dataset)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

poly_reg = PolynomialFeatures(degree=3)
x_poly = poly_reg.fit_transform(x)

pol_reg = LinearRegression()
pol_reg.fit(x_poly, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1_range = np.linspace(min(x[:, 0]), max(x[:, 0]), 30)
x2_range = np.linspace(min(x[:, 1]), max(x[:, 1]), 30)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

grid_data = np.c_[x1_grid.ravel(), x2_grid.ravel()]
grid_poly = poly_reg.transform(grid_data)
predictions = pol_reg.predict(grid_poly).reshape(x1_grid.shape)

ax.scatter(x[:, 0], x[:, 1], y, color='red')
ax.plot_surface(x1_grid, x2_grid, predictions, color='blue', alpha=0.6)

ax.set_title('Polynomial Regression with Two Independent Variables')
ax.set_xlabel('Variable 1')
ax.set_ylabel('Variable 2')
ax.set_zlabel('Dependent Variable')
plt.show()

print("R2 Score: ", pol_reg.score(x_poly, y))