import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 200m event at Asian Games with PB in position
filtered_df = df[(df['Event'] == '200 m') & (df['Competition'] == 'Asian Games') & (df['Position'].str.contains('PB', na=False))]
# Extract the year
year_pb = filtered_df['Year'].iloc[0]
print(f"Final Answer: {year_pb}")