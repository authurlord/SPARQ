import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'year' to integer for proper plotting
df['year'] = pd.to_numeric(df['year'])

# Plot the trend of qualifying speed over the years
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['qual'], marker='o', linestyle='-', color='b')
plt.title('Trend of Qualifying Speed (qual) from 1960 to 1971')
plt.xlabel('Year')
plt.ylabel('Qualifying Speed (mph)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Analyze the trend
trend = "increasing" if df['qual'].iloc[-1] > df['qual'].iloc[0] else "decreasing"
print(f"Final Answer: {trend}")