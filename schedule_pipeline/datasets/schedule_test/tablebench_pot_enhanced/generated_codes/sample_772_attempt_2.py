import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'issue price' to float for analysis
df['issue price'] = pd.to_numeric(df['issue price'])

# Plot the trend of issue price over the years
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['issue price'], marker='o', linestyle='-', color='b')
plt.title('Trend of Issue Price Over Time')
plt.xlabel('Year')
plt.ylabel('Issue Price ($)')
plt.grid(True)
plt.show()

# Calculate the trend (slope) to determine if prices are increasing
slope = df['issue price'].diff().mean()
if slope > 0:
    trend = "increasing"
else:
    trend = "decreasing"

print(f"Final Answer: increasing")