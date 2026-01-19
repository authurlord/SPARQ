import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 200m event at Asian Games with PB
pb_row = df[(df['Event'] == '200 m') & (df['Competition'] == 'Asian Games') & (df['Position'] == 'SF1–1st PB')]
# Extract the year
year_pb = pb_row['Year'].iloc[0]
print(f"Final Answer: {year_pb}")