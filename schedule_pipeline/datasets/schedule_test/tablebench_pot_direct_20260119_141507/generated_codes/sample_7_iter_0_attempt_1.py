import pandas as pd

df = pd.read_csv('table.csv')
# Filter rivers with 'left' position and sum their lengths
left_rivers_length = df[df['Position'] == 'left']['Length [km]'].sum()
print(f"Final Answer: {left_rivers_length}")