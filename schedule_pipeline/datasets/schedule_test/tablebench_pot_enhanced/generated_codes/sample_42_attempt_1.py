import pandas as pd

df = pd.read_csv('table.csv')
# Filter for wheat
wheat_data = df[df['commodity'] == 'wheat']
# Select the required years
years = ['2002 - 03', '2003 - 04', '2004 - 05', '2005 - 06']
wheat_values = wheat_data[years].astype(int).mean().mean()
print(f"Final Answer: {wheat_values:.1f}")