import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'total revenue' to numeric
df['total revenue'] = pd.to_numeric(df['total revenue'])

# Extract year and total revenue for plotting and projection
years = df['year'].astype(int)
revenue = df['total revenue']

# Fit a linear trend line
slope, intercept = np.polyfit(years, revenue, 1)
projected_revenue = slope * (years[-1] + 1) + intercept

# Plot the trend
plt.plot(years, revenue, marker='o', label='Total Revenue')
plt.plot([years[-1], years[-1] + 1], [revenue[-1], projected_revenue], color='red', linestyle='--', label='Projected')
plt.title('Trend in Total Revenue Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.legend()
plt.grid(True)
plt.show()

print(f"Final Answer: increasing, {int(projected_revenue)}")