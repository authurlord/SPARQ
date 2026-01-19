import pandas as pd

df = pd.read_csv('table.csv')
# Find the district with the highest population density
max_density = df.loc[df['pop density (per km2)'].idxmax()]
print(f"Final Answer: {max_density['district']}")