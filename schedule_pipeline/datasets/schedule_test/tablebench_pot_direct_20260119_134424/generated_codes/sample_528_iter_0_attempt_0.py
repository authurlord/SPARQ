import pandas as pd

df = pd.read_csv('table.csv')

# Extract seat counts for Kadets in First and Second Dumas
first_duma_kadets = int(df[df['Party'] == 'Constitutional Democratic Party (Kadets)']['First Duma'].values[0])
second_duma_kadets = int(df[df['Party'] == 'Constitutional Democratic Party (Kadets)']['Second Duma'].values[0])

# Total seats in each Duma
total_first_duma = int(df[df['Party'] == 'TOTAL']['First Duma'].values[0])
total_second_duma = int(df[df['Party'] == 'TOTAL']['Second Duma'].values[0])

# Calculate seat share percentages
first_share = (first_duma_kadets / total_first_duma) * 100
second_share = (second_duma_kadets / total_second_duma) * 100

# Calculate seat share change
seat_share_change = second_share - first_share

print(f"Final Answer: {seat_share_change:.1f}")