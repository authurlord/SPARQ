import pandas as pd

df = pd.read_csv('table.csv')
# Find the year where winnings are 411728
year_with_winnings = df[df['winnings'] == 411728]['year'].iloc[0]
print(f"Final Answer: {year_with_winnings}")