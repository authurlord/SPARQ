import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Asian Games and 200m event with PB
pb_row = df[(df['Event'] == '200 m') & (df['Competition'] == 'Asian Games') & (df['Position'].str.contains('PB', na=False))]
# Extract the year
year_pb = pb_row['Year'].iloc[0]
print(f"Final Answer: {year_pb}")