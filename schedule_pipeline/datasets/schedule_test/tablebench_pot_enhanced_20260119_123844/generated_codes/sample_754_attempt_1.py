import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'qual' and 'finish' to numeric
df['qual'] = pd.to_numeric(df['qual'], errors='coerce')
df['finish'] = pd.to_numeric(df['finish'], errors='coerce')

# Plot the trend of qualifying speed over the years
plt.figure(figsize=(10, 5))
plt.plot(df['year'], df['qual'], marker='o', label='Qualifying Speed (mph)')
plt.title('Trend of Qualifying Speed (1960–1971)')
plt.xlabel('Year')
plt.ylabel('Qualifying Speed (mph)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute correlation between qualifying speed and finishing position
correlation = df['qual'].corr(df['finish'])

# Interpretation: if correlation is negative, higher qualifying speed correlates with better (lower) finish
if correlation < 0:
    trend = "improving"
else:
    trend = "not clearly improving"

print(f"Final Answer: increasing, {trend}")