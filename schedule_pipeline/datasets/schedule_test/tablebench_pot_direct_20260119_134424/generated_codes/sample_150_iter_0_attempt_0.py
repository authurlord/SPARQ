import pandas as pd

df = pd.read_csv('table.csv')
# Extract population in 1939 and 2008
pop_1939 = df[df['year (january)'] == '1939']['population (000)'].values[0]
pop_2008 = df[df['year (january)'] == '2008']['population (000)'].values[0]

# Calculate growth rate
growth_rate = ((int(pop_2008) - int(pop_1939)) / int(pop_1939)) * 100

print(f"Final Answer: {growth_rate:.1f}")