import pandas as pd

df = pd.read_csv('table.csv')

# Extract numeric part from 'US Chart position', handle non-numeric by converting to NaN
df['US Chart position'] = df['US Chart position'].astype(str).str.extract('(\d+)').astype(float)

# Find the year with the highest and lowest US chart position
max_position_year = df.loc[df['US Chart position'].idxmax(), 'Year']
min_position_year = df.loc[df['US Chart position'].idxmin(), 'Year']

print(f"Final Answer: {max_position_year}, {min_position_year}")