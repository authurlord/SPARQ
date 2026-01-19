import pandas as pd

df = pd.read_csv('table.csv')
# Filter for CARIFTA Games (U-20), 400 m, 1st position, and PB in notes
filtered_df = df[
    (df['Competition'] == 'CARIFTA Games (U-20)') &
    (df['Event'] == '400 m') &
    (df['Position'] == '1st') &
    (df['Notes'].str.contains('PB', na=False))
]
# Get the year from the filtered result
year = filtered_df['Year'].iloc[0]
print(f"Final Answer: {year}")