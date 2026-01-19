import pandas as pd

df = pd.read_csv('table.csv')
# Find the crater with the largest diameter
max_diameter_row = df.loc[df['diameter (km)'].idxmax()]
largest_crater = max_diameter_row['name']
print(f"Final Answer: {largest_crater}")