import pandas as pd

df = pd.read_csv('table.csv')

# Extract population in 1939 and 2008
pop_1939 = int(df[df['year (january)'] == '1939']['population (000)'].iloc[0])
pop_2008 = int(df[df['year (january)'] == '2008']['population (000)'].iloc[0])

# Calculate population growth rate
growth_rate = ((pop_2008 - pop_1939) / pop_1939) * 100

# Urban percentage change
urban_1939 = int(df[df['year (january)'] == '1939']['urban , %'].iloc[0])
urban_2008 = int(df[df['year (january)'] == '2008']['urban , %'].iloc[0])

# Output the results
print(f"Final Answer: {growth_rate:.1f}, {urban_2008 - urban_1939}")