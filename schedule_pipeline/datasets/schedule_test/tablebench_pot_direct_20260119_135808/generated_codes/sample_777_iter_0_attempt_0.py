import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'total revenue' to numeric
df['total revenue'] = pd.to_numeric(df['total revenue'])

# Plot the trend of total revenue over the years
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['total revenue'], marker='o', linestyle='-', color='b')
plt.title('Trend of Total Revenue Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Determine the trend
if df['total revenue'].iloc[-1] > df['total revenue'].iloc[0]:
    trend = "increasing"
elif df['total revenue'].iloc[-1] < df['total revenue'].iloc[0]:
    trend = "decreasing"
else:
    trend = "stable"

print(f"Final Answer: {trend}")