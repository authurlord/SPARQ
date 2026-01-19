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

# The issue price shows a consistent upward trend
print("Final Answer: increasing")