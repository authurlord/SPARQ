import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with the maximum diameter
max_diameter_row = df.loc[df['diameter (km)'].idxmax()]
crater_name = max_diameter_row['name']
print(f"Final Answer: {crater_name}")