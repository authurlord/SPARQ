import pandas as pd

df = pd.read_csv('table.csv')

# Extract population in 1939 and 2008
pop_1939 = df[df['year (january)'] == '1939']['population (000)'].values[0]
pop_2008 = df[df['year (january)'] == '2008']['population (000)'].values[0]

# Calculate growth rate
growth_rate = ((float(pop_2008) - float(pop_1939)) / float(pop_1939)) * 100

# Urban percentage change
urban_1939 = df[df['year (january)'] == '1939']['urban , %'].values[0]
urban_2008 = df[df['year (january)'] == '2008']['urban , %'].values[0]

urban_change = int(urban_2008) - int(urban_1939)

# Output final answer based on growth rate and urbanization trend
print(f"Final Answer: {growth_rate:.1f}, {urban_change}")