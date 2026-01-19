import pandas as pd

df = pd.read_csv('table.csv')
# Find the row for 'Constitutional Democratic Party (Kadets)'
kadet_row = df[df['Party'] == 'Constitutional Democratic Party (Kadets)']
# Extract seat counts for First and Second Duma
first_duma_seats = int(kadet_row['First Duma'].values[0].split()[0])
second_duma_seats = int(kadet_row['Second Duma'].values[0].split()[0])
# Calculate the change
seat_change = second_duma_seats - first_duma_seats
print(f"Final Answer: {seat_change}")