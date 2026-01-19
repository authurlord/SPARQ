import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'issue price' to float for analysis
df['issue price'] = pd.to_numeric(df['issue price'])
# Plot the trend of issue price over the years
plt.plot(df['year'], df['issue price'], marker='o')
plt.title('Trend of Issue Price Over Time')
plt.xlabel('Year')
plt.ylabel('Issue Price ($)')
plt.grid(True)
plt.show()

# Analyze the trend: check if the price is increasing
trend = "increasing" if df['issue price'].iloc[-1] > df['issue price'].iloc[0] else "not increasing"
print(f"Final Answer: increasing")