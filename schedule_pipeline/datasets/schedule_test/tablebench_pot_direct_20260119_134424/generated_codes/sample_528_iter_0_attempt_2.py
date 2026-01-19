import pandas as pd

df = pd.read_csv('table.csv')
# Extract seat counts for Constitutional Democratic Party (Kadets)
first_duma_kadets = int(df[df['Party'] == 'Constitutional Democratic Party (Kadets)']['First Duma'].values[0])
second_duma_kadets = int(df[df['Party'] == 'Constitutional Democratic Party (Kadets)']['Second Duma'].values[0])

# Calculate percentage change
seat_share_change = ((second_duma_kadets - first_duma_kadets) / first_duma_kadets) * 100
print(f"Final Answer: {seat_share_change:.1f}")