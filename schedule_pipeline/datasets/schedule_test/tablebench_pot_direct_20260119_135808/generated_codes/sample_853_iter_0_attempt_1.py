import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'US Chart position' column by extracting numeric values
df['US Chart position'] = df['US Chart position'].astype(str).str.extract('(\d+)').astype(float)

# Find the year with the highest US chart position (highest number means worst position)
max_position_year = df.loc[df['US Chart position'].idxmax(), 'Year']

# Find the year with the lowest US chart position (lowest number means best position)
min_position_year = df.loc[df['US Chart position'].idxmin(), 'Year']

print(f"Final Answer: {max_position_year}, {min_position_year}")