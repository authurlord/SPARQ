import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Position is 'left'
left_rivers = df[df['Position'] == 'left']
# Convert 'Length [km]' to numeric and sum
total_length_left = left_rivers['Length [km]'].astype(float).sum()
print(f"Final Answer: {total_length_left}")