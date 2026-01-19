import pandas as pd

df = pd.read_csv('table.csv')
# Extract seat numbers for Constitutional Democratic Party (Kadets)
first_duma_kadets = int(df[df['Party'] == 'Constitutional Democratic Party (Kadets)']['First Duma'].iloc[0])
second_duma_kadets = int(df[df['Party'] == 'Constitutional Democratic Party (Kadets)']['Second Duma'].iloc[0])

# Calculate the change in seats
seat_change = second_duma_kadets - first_duma_kadets
print(f"Final Answer: {seat_change}")