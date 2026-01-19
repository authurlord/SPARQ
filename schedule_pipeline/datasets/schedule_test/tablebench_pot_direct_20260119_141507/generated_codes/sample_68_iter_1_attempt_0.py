import pandas as pd

df = pd.read_csv('table.csv')
# Find the year where winnings are 411728
target_winnings = 411728
result_year = df[df['winnings'] == target_winnings]['year'].values[0]
print(f"Final Answer: {result_year}")