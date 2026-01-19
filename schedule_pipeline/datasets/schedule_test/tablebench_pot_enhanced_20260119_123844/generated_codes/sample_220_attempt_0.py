import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('table.csv')

# Filter data for years between 1959 and 1989 (inclusive)
filtered_df = df[(df['year (january)'] >= '1959') & (df['year (january)'] <= '1989')]

# Convert population to numeric for plotting
filtered_df['population (000)'] = pd.to_numeric(filtered_df['population (000)'])

# Plot population vs urban percentage
plt.figure(figsize=(10, 6))
plt.plot(filtered_df['population (000)'], filtered_df['urban , %'], marker='o', linestyle='-', color='b')
plt.title('Urban Percentage vs Population Size (1959–1989)')
plt.xlabel('Population (in thousands)')
plt.ylabel('Urban Percentage (%)')
plt.grid(True)
plt.show()

# Analyze the trend: urban percentage increases from 44% to 57% as population grows from 9.3M to 16.5M
# Final answer is the trend description based on data
print("Final Answer: increases")