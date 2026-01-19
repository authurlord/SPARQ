import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows based on conditions
filtered_df = df[
    (df['Event'] == '400 m') &
    (df['Competition'] == 'CARIFTA Games (U-20)') &
    (df['Position'] == '1st') &
    (df['Notes'].str.contains('PB', case=False))
]

# Get the year of the first match (only one such row expected)
year = filtered_df.iloc[0]['Year'] if not filtered_df.empty else None
print(f"Final Answer: {year}")