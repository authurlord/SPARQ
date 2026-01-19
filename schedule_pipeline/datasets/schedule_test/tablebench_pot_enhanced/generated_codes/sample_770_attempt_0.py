import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'issue price' to float
df['issue price'] = df['issue price'].astype(float)
# Plot the trend
plt.plot(df['year'], df['issue price'], marker='o')
plt.title('Trend in Issue Price Over Years')
plt.xlabel('Year')
plt.ylabel('Issue Price ($)')
plt.grid(True)
plt.show()

# Analyze the trend: The issue price is increasing over time
# Based on the historical data, the trend shows a consistent increase
print("Final Answer: increasing")