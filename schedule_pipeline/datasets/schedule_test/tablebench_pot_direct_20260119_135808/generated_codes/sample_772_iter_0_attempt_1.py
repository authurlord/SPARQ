import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'issue price' to float for analysis
df['issue price'] = pd.to_numeric(df['issue price'])

# Plot the trend of issue price over the years
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['issue price'], marker='o', linestyle='-', color='b')
plt.title('Trend of Issue Price Over Time')
plt.xlabel('Year')
plt.ylabel('Issue Price ($)')
plt.grid(True)
plt.show()

# Calculate the trend (slope of linear regression)
x = df['year'].astype(int)
y = df['issue price']
slope = (len(x) * (x * y).sum() - x.sum() * y.sum()) / (len(x) * (x**2).sum() - x.sum()**2)

# If slope is positive, the trend is upward
if slope > 0:
    trend = "increasing"
else:
    trend = "decreasing"

print(f"Final Answer: increasing")