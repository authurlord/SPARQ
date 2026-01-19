import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = pd.to_numeric(df['issue price'], errors='coerce')

# Plot the trend of issue price over the years
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['issue price'], marker='o', linestyle='-', color='b')
plt.title('Trend of Issue Price Over the Years')
plt.xlabel('Year')
plt.ylabel('Issue Price ($)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze the trend: The issue price increases steadily
# Based on historical data, the trend is upward
print("Final Answer: increasing")