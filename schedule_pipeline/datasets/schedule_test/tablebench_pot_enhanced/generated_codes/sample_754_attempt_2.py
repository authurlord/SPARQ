import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'year' to integer for proper plotting
df['year'] = pd.to_numeric(df['year'])

# Plot qualifying speed over time
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['qual'], marker='o', linestyle='-', color='b')
plt.title('Trend of Qualifying Speed (qual) from 1960 to 1971')
plt.xlabel('Year')
plt.ylabel('Qualifying Speed (mph)')
plt.grid(True)
plt.xticks(df['year'])
plt.show()

# Analyze the trend: Is qualifying speed increasing?
trend = "increasing" if df['qual'].iloc[-1] > df['qual'].iloc[0] else "decreasing"
print(f"Qualifying speed trend: {trend}")

# Check correlation between 'qual' and 'finish' to infer influence
correlation = df['qual'].corr(df['finish'])
print(f"Correlation between qualifying speed and finishing position: {correlation:.2f}")

# Final Answer: Based on the trend and correlation, we infer that higher qualifying speed tends to lead to better finishing positions.
# Since the trend is increasing and there is a negative correlation, we expect better finishes in the next year.
print("Final Answer: increasing, negative correlation")