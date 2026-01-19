import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')

# Convert 'bötzow' column to float
df['bötzow'] = df['bötzow'].astype(float)

# Prepare data for regression
X = df['year'].values.reshape(-1, 1)
y = df['bötzow'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for 2015
predicted_2015 = model.predict([[2015]])[0]

# Plot the trend
plt.plot(df['year'], df['bötzow'], marker='o', label='Actual')
plt.axvline(x=2010, color='r', linestyle='--', label='Last known data')
plt.plot(2015, predicted_2015, marker='s', color='g', label='Predicted 2015')
plt.xlabel('Year')
plt.ylabel('Population (in thousands)')
plt.title('Population Trend of Bötzow and Prediction for 2015')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final Answer: {predicted_2015:.3f}")