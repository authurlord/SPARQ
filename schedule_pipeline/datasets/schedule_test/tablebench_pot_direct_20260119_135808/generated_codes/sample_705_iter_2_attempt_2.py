import pandas as pd

df = pd.read_csv('table.csv')
# Drop the first row which contains metadata
df = df.drop(df.index[0])
# Find the row with the maximum Copper (mg)
max_copper_row = df.loc[df['Copper (mg)'].idxmax()]
# Extract the staple food (column name) with the highest Copper (mg)
staple_with_max_copper = max_copper_row.name
print(f"Final Answer: {staple_with_max_copper}")