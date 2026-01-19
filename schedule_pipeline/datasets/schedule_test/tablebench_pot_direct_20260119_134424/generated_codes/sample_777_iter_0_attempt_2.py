import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'total revenue' to numeric
df['total revenue'] = pd.to_numeric(df['total revenue'])
# Plot the trend
plt.plot(df['year'], df['total revenue'], marker='o')
plt.title('Trend of Total Revenue Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
# Final answer is the trend description based on the plot
print("Final Answer: increasing")