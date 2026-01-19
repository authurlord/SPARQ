import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'diameter (km)' to numeric for proper comparison
df['diameter (km)'] = pd.to_numeric(df['diameter (km)'])
# Find the row with the maximum diameter
max_diameter_crater = df.loc[df['diameter (km)'].idxmax()]
# Extract the name of the crater
crater_name = max_diameter_crater['name']
print(f"Final Answer: {crater_name}")