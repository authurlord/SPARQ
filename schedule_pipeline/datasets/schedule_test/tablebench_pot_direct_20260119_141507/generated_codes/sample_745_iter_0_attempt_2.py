import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for the year 2002 (last active year)
row_2002 = df[df['year'] == '2002'].iloc[0]
avg_finish = row_2002['avg finish']
total_winnings = row_2002['winnings']

print(f"Final Answer: {avg_finish}, {total_winnings}")