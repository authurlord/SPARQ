import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the maximum diameter
max_diameter_crater = df.loc[df['diameter (km)'].idxmax()]
print(f"Final Answer: {max_diameter_crater['name']}")