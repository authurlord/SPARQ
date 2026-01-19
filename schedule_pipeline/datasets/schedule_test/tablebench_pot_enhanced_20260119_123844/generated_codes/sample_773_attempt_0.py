import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'total revenue' to numeric
df['total revenue'] = pd.to_numeric(df['total revenue'])

# Plot the trend
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['total revenue'], marker='o', linestyle='-', color='b')
plt.title('Trend in Total Revenue Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.show()

# Fit a linear regression model to project next year's revenue
X = np.array(df['year'].astype(int)).reshape(-1, 1)
y = df['total revenue'].values
model = np.polyfit(X.flatten(), y, 1)
projected_revenue = np.polyval(model, int(df['year'].iloc[-1]) + 1)

print(f"Final Answer: increasing, {int(projected_revenue)}")