import pandas as pd

df = pd.read_csv('table.csv')
# Extract seats for Kadets in First and Second Duma
kadets_first_duma = int(df[df['Party'] == 'Constitutional Democratic Party (Kadets)']['First Duma'].values[0])
kadets_second_duma = int(df[df['Party'] == 'Constitutional Democratic Party (Kadets)']['Second Duma'].values[0])

# Total seats in First and Second Duma
total_first_duma = int(df[df['Party'] == 'TOTAL']['First Duma'].values[0])
total_second_duma = int(df[df['Party'] == 'TOTAL']['Second Duma'].values[0])

# Calculate seat share percentages
share_first = (kadets_first_duma / total_first_duma) * 100
share_second = (kadets_second_duma / total_second_duma) * 100

# Calculate change in seat share
seat_share_change = share_second - share_first

print(f"Final Answer: {seat_share_change:.1f}")