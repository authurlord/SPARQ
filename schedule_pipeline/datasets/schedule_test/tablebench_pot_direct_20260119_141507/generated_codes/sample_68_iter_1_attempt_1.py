import pandas as pd

df = pd.read_csv('table.csv')

# Convert winnings column to numeric, replacing commas and parsing as float
df['winnings'] = df['winnings'].str.replace(',', '').astype(float)

# Find the year where winnings equal 411728
target_winnings = 411728
result_year = df[df['winnings'] == target_winnings]['year'].values[0]

print(f"Final Answer: {result_year}")