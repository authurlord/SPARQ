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
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Perform linear regression to project next year's revenue
X = np.array(df['year'].astype(int))
y = df['total revenue'].values
slope, intercept = np.polyfit(X, y, 1)
projected_year = int(df['year'].iloc[-1]) + 1
projected_revenue = slope * projected_year + intercept

print(f"Final Answer: Increasing trend, {projected_revenue:.0f}")