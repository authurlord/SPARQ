import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'total revenue' to numeric
df['total revenue'] = pd.to_numeric(df['total revenue'])

# Extract year and total revenue for modeling
X = df['year'].values.reshape(-1, 1)
y = df['total revenue'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict next year (2006)
next_year = np.array([[2006]])
predicted_revenue = model.predict(next_year)[0]

# Plot the trend
plt.plot(df['year'], df['total revenue'], marker='o')
plt.title('Total Revenue Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.show()

print(f"Final Answer: Increasing trend, {predicted_revenue:.0f}")