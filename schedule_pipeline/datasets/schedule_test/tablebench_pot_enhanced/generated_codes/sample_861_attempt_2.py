import pandas as pd

df = pd.read_csv('table.csv')
# Remove the total row
df = df[df['incident no'] != 'total']
# Convert 'killed' to integer for comparison
df['killed'] = pd.to_numeric(df['killed'])
# Find the place with the highest number of killed
max_killed_place = df.loc[df['killed'].idxmax(), 'place']
print(f"Final Answer: {max_killed_place}")