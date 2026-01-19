import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter data for years between 1959 and 1989
filtered_df = df[(df['year (january)'].astype(int) >= 1959) & (df['year (january)'].astype(int) <= 1989)]

# Convert columns to numeric
filtered_df['population (000)'] = pd.to_numeric(filtered_df['population (000)'])
filtered_df['urban , %'] = pd.to_numeric(filtered_df['urban , %'])

# Plot urban percentage vs population size
plt.figure(figsize=(10, 6))
plt.plot(filtered_df['population (000)'], filtered_df['urban , %'], marker='o', linestyle='-', color='b')
plt.title('Urban Percentage vs Population Size (1959–1989)')
plt.xlabel('Population (in thousands)')
plt.ylabel('Urban Percentage (%)')
plt.grid(True)
plt.show()

# Final answer based on trend: urban percentage increases as population increases
print("Final Answer: increases")