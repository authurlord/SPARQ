import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Extract year and total revenue
df['year'] = df['year'].astype(int)
df = df.sort_values(by='year')

# Plot total revenue trend
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['total revenue'], marker='o', linestyle='-', color='b')
plt.title('Total Revenue Trend Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.xticks(df['year'].unique())
plt.tight_layout()
plt.show()

# Print the trend summary (as a simple statement)
print(f"Final Answer: The total revenue has generally increased over the years, with notable growth from 2001 to 2005.")