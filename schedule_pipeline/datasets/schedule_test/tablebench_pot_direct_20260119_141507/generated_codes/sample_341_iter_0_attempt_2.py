import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is '400 m' and Competition is 'CARIFTA Games (U-20)'
filtered_df = df[(df['Event'] == '400 m') & (df['Competition'] == 'CARIFTA Games (U-20)')]
# Further filter where Notes contains 'PB' (personal best)
pb_filtered = filtered_df[filtered_df['Notes'].str.contains('PB', case=False)]
# Extract the Year
year = pb_filtered.iloc[0]['Year'] if not pb_filtered.empty else None
print(f"Final Answer: {year}")