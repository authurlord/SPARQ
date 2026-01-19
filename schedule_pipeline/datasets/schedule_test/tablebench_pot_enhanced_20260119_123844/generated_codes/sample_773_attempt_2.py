import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('table.csv')
# Convert 'total revenue' to numeric
df['total revenue'] = pd.to_numeric(df['total revenue'])

# Extract year and total revenue for analysis
years = df['year'].astype(int).values.reshape(-1, 1)
revenue = df['total revenue'].values

# Fit a linear regression model
model = LinearRegression()
model.fit(years, revenue)

# Predict next year's revenue
next_year = np.array([[2006]])
predicted_revenue = model.predict(next_year)[0]

# Plot the trend
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['total revenue'], marker='o', label='Total Revenue')
plt.title('Trend in Total Revenue Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.xticks(df['year'])
plt.legend()
plt.grid(True)
plt.show()

print(f"Final Answer: increasing, {int(predicted_revenue)}")