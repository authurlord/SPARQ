import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is '3000 m steeplechase' and Position is '1st'
filtered_df = df[(df['Event'] == '3000 m steeplechase') & (df['Position'] == '1st')]
# Check if the Venue is Nassau, Bahamas (capital of Bahamas)
if not filtered_df.empty and 'Nassau, Bahamas' in filtered_df['Venue'].values:
    competition = filtered_df.iloc[0]['Competition']
    print(f"Final Answer: {competition}")
else:
    print("Final Answer: None")