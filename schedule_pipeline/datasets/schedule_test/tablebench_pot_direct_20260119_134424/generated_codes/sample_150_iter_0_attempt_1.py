import pandas as pd

df = pd.read_csv('table.csv')

# Extract initial and final population
initial_pop = df[df['year (january)'] == '1939']['population (000)'].values[0]
final_pop = df[df['year (january)'] == '2008']['population (000)'].values[0]

# Calculate population growth rate
growth_rate = ((int(final_pop) - int(initial_pop)) / int(initial_pop)) * 100

# Urban percentage change
initial_urban = df[df['year (january)'] == '1939']['urban , %'].values[0]
final_urban = df[df['year (january)'] == '2008']['urban , %'].values[0]

# Output results
print(f"Population growth rate (1939–2008): {growth_rate:.1f}%")
print(f"Urban percentage change: {initial_urban}% → {final_urban}%")

# Final answer based on observation: urbanization increased significantly while population grew
Final Answer: urbanization contributed to population growth, 1939–2008