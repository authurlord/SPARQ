import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Position is 'left'
left_rivers = df[df['Position'] == 'left']
# Sum the Length [km] column
total_length_left = left_rivers['Length [km]'].sum()
print(f"Final Answer: {total_length_left}")