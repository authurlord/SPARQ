import pandas as pd

df = pd.read_csv('table.csv')
# Filter the row for the last active year (2002)
last_active_row = df[df['year'] == '2002'].iloc[0]
average_finish_position = last_active_row['avg finish']
total_winnings = last_active_row['winnings']
print(f"Final Answer: {average_finish_position}, {total_winnings}")