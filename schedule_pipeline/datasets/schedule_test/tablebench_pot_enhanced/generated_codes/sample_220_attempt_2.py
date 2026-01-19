import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter data between 1959 and 1989
filtered_df = df[(df['year (january)'] >= '1959') & (df['year (january)'] <= '1989')]
# Convert population to numeric for plotting
filtered_df['population (000)'] = pd.to_numeric(filtered_df['population (000)'])
# Plot scatter plot
plt.scatter(filtered_df['population (000)'], filtered_df['urban , %'])
plt.xlabel('Population (000)')
plt.ylabel('Urban Percentage (%)')
plt.title('Urban Percentage vs Population (1959-1989)')
plt.grid(True)
plt.show()
# Final answer based on trend: urban percentage increases with population
print("Final Answer: increases")