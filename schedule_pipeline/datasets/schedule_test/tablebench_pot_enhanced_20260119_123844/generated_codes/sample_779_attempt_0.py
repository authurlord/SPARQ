import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('table.csv')

# Convert 'bötzow' column to float
df['bötzow'] = pd.to_numeric(df['bötzow'])

# Prepare data for regression
X = df['year'].values.reshape(-1, 1)  # Years as features
y = df['bötzow'].values  # Population as target

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict population for 2015
prediction_2015 = model.predict([[2015]])

# Plot the trend
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['bötzow'], marker='o', label='Bötzow Population')
plt.axvline(x=2010, color='r', linestyle='--', label='Last Data Point')
plt.title('Population Trend of Bötzow Over the Years')
plt.xlabel('Year')
plt.ylabel('Population (in thousands)')
plt.legend()
plt.grid(True)
plt.show()

# Print the predicted value
print(f"Final Answer: {prediction_2015[0]:.3f}")