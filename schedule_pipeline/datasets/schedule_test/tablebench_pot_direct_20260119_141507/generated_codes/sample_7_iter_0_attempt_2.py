import pandas as pd

df = pd.read_csv('table.csv')
# Filter rivers that flow into the left side
left_rivers = df[df['Position'] == 'left']
# Sum the lengths of these rivers
total_length_left = left_rivers['Length [km]'].sum()
print(f"Final Answer: {total_length_left}")