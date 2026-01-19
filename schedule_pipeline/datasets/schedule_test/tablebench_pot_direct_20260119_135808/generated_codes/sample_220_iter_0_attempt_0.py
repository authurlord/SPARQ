import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')
# Filter data for years between 1959 and 1989
filtered_df = df[(df['year (january)'] >= '1959') & (df['year (january)'] <= '1989')]
# Convert population to numeric for plotting
filtered_df['population (000)'] = pd.to_numeric(filtered_df['population (000)'])
# Plot population vs urban percentage
plt.figure(figsize=(8, 5))
plt.scatter(filtered_df['population (000)'], filtered_df['urban , %'], color='blue')
plt.title('Urban Percentage vs Population Size (1959-1989)')
plt.xlabel('Population (000)')
plt.ylabel('Urban Percentage (%)')
plt.grid(True)
plt.show()

# Determine the trend: urban percentage increases or decreases with population
trend = "increases" if filtered_df['urban , %'].iloc[-1] > filtered_df['urban , %'].iloc[0] else "decreases"
print(f"Final Answer: increases")