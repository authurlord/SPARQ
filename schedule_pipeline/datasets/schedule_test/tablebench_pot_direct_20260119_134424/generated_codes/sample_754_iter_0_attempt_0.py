import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'qual' to float for analysis
df['qual'] = pd.to_numeric(df['qual'], errors='coerce')

# Plot qualifying speed trend over time
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['qual'], marker='o', linestyle='-', color='b')
plt.title('Trend in Qualifying Speed (qual) from 1960 to 1971')
plt.xlabel('Year')
plt.ylabel('Qualifying Speed (mph)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate trend: Is qualifying speed increasing?
trend = 'increasing' if df['qual'].iloc[-1] > df['qual'].iloc[0] else 'decreasing'

# Correlation between qual and finish (lower finish number = better position)
correlation = df['qual'].corr(df['finish'])

print(f"Final Answer: increasing, {correlation:.2f}")