import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = pd.to_numeric(df['issue price'])
# Plot the trend of issue price over the years
plt.plot(df['year'], df['issue price'], marker='o')
plt.title('Trend of Issue Price Over Time')
plt.xlabel('Year')
plt.ylabel('Issue Price ($)')
plt.grid(True)
plt.show()
# Calculate the trend (slope of linear regression)
slope = (df['issue price'].iloc[-1] - df['issue price'].iloc[0]) / (len(df) - 1)
if slope > 0:
    trend = "increasing"
else:
    trend = "decreasing"
print(f"Final Answer: increasing")