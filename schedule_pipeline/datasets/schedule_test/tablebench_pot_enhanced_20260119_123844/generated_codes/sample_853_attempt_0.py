import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'US Chart position' to numeric, coercing errors to NaN
df['US Chart position'] = pd.to_numeric(df['US Chart position'], errors='coerce')

# Find the year with the highest US chart position (lowest number)
highest_position_year = df.loc[df['US Chart position'].idxmin(), 'Year']

# Find the year with the lowest US chart position (highest number)
lowest_position_year = df.loc[df['US Chart position'].idxmax(), 'Year']

print(f"Final Answer: {highest_position_year}, {lowest_position_year}")