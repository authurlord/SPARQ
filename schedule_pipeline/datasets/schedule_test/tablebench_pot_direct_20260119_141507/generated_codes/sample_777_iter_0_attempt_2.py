import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert year to integer for proper sorting
df['year'] = pd.to_numeric(df['year'], errors='coerce')
# Sort by year
df = df.sort_values(by='year')
# Plot total revenue over years
plt.figure(figsize=(10, 6))
plt.plot(df['year'], df['total revenue'], marker='o', linestyle='-', color='b')
plt.title('Total Revenue Trend Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.xticks(df['year'].unique())
plt.tight_layout()
plt.show()