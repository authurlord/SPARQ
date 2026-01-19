import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'qual' to numeric
df['qual'] = pd.to_numeric(df['qual'])

# Plot the trend of qualifying speed over the years
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['qual'], marker='o', linestyle='-', color='b')
plt.title('Trend of Qualifying Speed (1960–1971)')
plt.xlabel('Year')
plt.ylabel('Qualifying Speed (mph)')
plt.grid(True)
plt.show()

# Analyze trend: if qualifying speed is increasing, it may lead to better finishing positions
trend = "increasing" if df['qual'].iloc[-1] > df['qual'].iloc[0] else "decreasing"
print(f"Final Answer: {trend}")