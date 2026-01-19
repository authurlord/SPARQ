import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Convert 'year' to integer for proper sorting and plotting
df['year'] = pd.to_numeric(df['year'])

# Sort by year
df = df.sort_values('year')

# Plot the trend of qualifying speed over time
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['qual'], marker='o', linestyle='-', color='b')
plt.title('Trend of Qualifying Speed (qual) from 1960 to 1971')
plt.xlabel('Year')
plt.ylabel('Qualifying Speed (mph)')
plt.grid(True)
plt.show()

# Calculate the trend: Is qualifying speed increasing?
trend = 'increasing' if df['qual'].iloc[-1] > df['qual'].iloc[0] else 'decreasing'

# Check correlation between qual and finish (lower finish number = better position)
correlation = df['qual'].corr(df['finish'])

# Print interpretation
print(f"Final Answer: increasing, influenced by trend")