import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'issue price' to numeric for analysis
df['issue price'] = pd.to_numeric(df['issue price'], errors='coerce')

# Plot the trend of issue price over the years
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['issue price'], marker='o', linestyle='-', color='b')
plt.title('Trend in Issue Price of Coins Over Time')
plt.xlabel('Year')
plt.ylabel('Issue Price ($)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze the trend: The issue price increases steadily over time
# From 2002 to 2012, the price increased from $39.95 to $49.95
# This suggests a consistent upward trend
# Extrapolating this trend, the price is expected to continue rising in the next 5 years
print("Final Answer: increasing")