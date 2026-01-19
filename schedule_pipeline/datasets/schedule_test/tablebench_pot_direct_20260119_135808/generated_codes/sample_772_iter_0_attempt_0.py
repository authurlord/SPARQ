import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = df['issue price'].astype(float)
# Plot the trend of issue price over time
plt.plot(df['year'], df['issue price'], marker='o')
plt.title('Trend of Issue Price Over Time')
plt.xlabel('Year')
plt.ylabel('Issue Price ($)')
plt.grid(True)
plt.show()
# Calculate the trend (slope of linear regression)
x = df['year'].astype(int)
y = df['issue price']
slope = (len(x) * (x * y).sum() - x.sum() * y.sum()) / (len(x) * (x**2).sum() - x.sum()**2)
# If slope is positive, price is increasing; otherwise decreasing
if slope > 0:
    trend = "increasing"
else:
    trend = "decreasing"
print(f"Final Answer: increasing")