import pandas as pd

df = pd.read_csv('table.csv')

# Extract population values for 1939 and 2008
initial_pop = df[df['year (january)'] == '1939']['population (000)'].values[0]
final_pop = df[df['year (january)'] == '2008']['population (000)'].values[0]

# Convert to numeric
initial_pop = int(initial_pop)
final_pop = int(final_pop)

# Calculate growth rate
growth_rate = ((final_pop - initial_pop) / initial_pop) * 100

print(f"Final Answer: {growth_rate:.1f}")