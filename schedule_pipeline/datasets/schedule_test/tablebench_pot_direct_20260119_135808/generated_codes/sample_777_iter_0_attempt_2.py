import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'total revenue' to numeric
df['total revenue'] = pd.to_numeric(df['total revenue'])

# Plot the trend
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['total revenue'], marker='o', linestyle='-', color='b')
plt.title('Trend of Total Revenue Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Final answer based on trend
print("Final Answer: increasing")