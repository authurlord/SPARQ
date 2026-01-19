import pandas as pd

df = pd.read_csv('table.csv')
# Find the season where 'giant slalom' is 7
season_7 = df[df['giant slalom'] == '7']['season'].values[0]
print(f"Final Answer: {season_7}")