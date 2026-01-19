import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'US Chart position' to numeric, coercing errors to NaN
df['US Chart position'] = pd.to_numeric(df['US Chart position'], errors='coerce')

# Find the year with the highest US chart position (lowest numerical value)
max_position_year = df.loc[df['US Chart position'].idxmax(), 'Year']

# Find the year with the lowest US chart position (highest numerical value)
min_position_year = df.loc[df['US Chart position'].idxmin(), 'Year']

print(f"Final Answer: {max_position_year}, {min_position_year}")