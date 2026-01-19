import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
df['year'] = pd.to_numeric(df['year'])
df['bötzow'] = pd.to_numeric(df['bötzow'])

# Prepare data for linear regression
X = df['year'].values.reshape(-1, 1)
y = df['bötzow'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict for 2015
predicted_2015 = model.predict([[2015]])[0]

# Plot the trend
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['bötzow'], marker='o', label='Actual Population')
plt.axvline(x=2010, color='gray', linestyle='--', label='Last Data Point')
plt.plot(2015, predicted_2015, marker='s', color='red', label='Predicted 2015')
plt.title('Population Trend of Bötzow Over the Years')
plt.xlabel('Year')
plt.ylabel('Population (in thousands)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final Answer: {predicted_2015:.3f}")