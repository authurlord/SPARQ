import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'total revenue' to numeric
df['total revenue'] = pd.to_numeric(df['total revenue'])

# Extract year and total revenue for analysis
years = df['year'].astype(int)
revenue = df['total revenue']

# Plot the trend
plt.figure(figsize=(10, 5))
plt.plot(years, revenue, marker='o', linestyle='-', color='b')
plt.title('Trend in Total Revenue Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.show()

# Perform linear regression to predict next year's revenue
X = years.values.reshape(-1, 1)
y = revenue.values
model = np.polyfit(X.flatten(), y, 1)
predicted_next_year = np.polyval(model, years.max() + 1)

print(f"Final Answer: increasing, {predicted_next_year:.0f}")