import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = pd.to_numeric(df['issue price'])

# Plot the trend of issue price over the years
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['issue price'], marker='o', linestyle='-', color='b')
plt.title('Trend of Issue Price Over Time')
plt.xlabel('Year')
plt.ylabel('Issue Price ($)')
plt.grid(True)
plt.show()

# Calculate the trend (slope) using linear regression
x = df['year'].astype(int)
y = df['issue price']
slope = (len(x) * (x * y).sum() - x.sum() * y.sum()) / (len(x) * (x**2).sum() - x.sum()**2)

# Determine the trend direction
if slope > 0:
    trend = "increasing"
elif slope < 0:
    trend = "decreasing"
else:
    trend = "stable"

print(f"Final Answer: increasing")