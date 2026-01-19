import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'foreign' column to numeric for proper comparison
df['foreign'] = pd.to_numeric(df['foreign'], errors='coerce')

# Find the municipality with the highest and lowest number of people speaking foreign
max_foreign_municipality = df.loc[df['foreign'].idxmax(), 'mapiri municipality']
min_foreign_municipality = df.loc[df['foreign'].idxmin(), 'mapiri municipality']

# Calculate the difference
difference = df['foreign'].max() - df['foreign'].min()

print(f"Final Answer: {max_foreign_municipality}, {difference:.0f}")