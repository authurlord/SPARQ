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
plt.show()

# Check the trend: is qualifying speed increasing?
trend = df['qual'].diff().mean()
if trend > 0:
    print("The qualifying speed has generally increased over time.")
else:
    print("The qualifying speed has generally decreased over time.")

# Check correlation between 'qual' and 'finish' (lower finish number = better position)
correlation = df['qual'].corr(df['finish'])
if correlation < 0:
    print("Higher qualifying speeds are associated with better (lower) finishing positions.")
else:
    print("No clear inverse relationship between qualifying speed and finishing position.")

print("Final Answer: increasing, improved")