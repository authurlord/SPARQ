import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Convert 'year' to integer for proper sorting and analysis
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# Sort by year to ensure chronological order
df = df.sort_values(by='year')

# Extract qualifying speed ('qual') over the years
qual_trend = df['qual'].values

# Plot the trend of qualifying speed
plt.figure(figsize=(10, 6))
plt.plot(df['year'], qual_trend, marker='o', linestyle='-', color='b')
plt.title('Trend of Qualifying Speed (1960–1971)')
plt.xlabel('Year')
plt.ylabel('Qualifying Speed (km/h)')
plt.grid(True)
plt.xticks(df['year'].unique())
plt.tight_layout()

# Show the plot
plt.show()

# Calculate the average qualifying speed
avg_qual_speed = df['qual'].mean()
print(f"Final Answer: increasing, might lead to better finishing positions")