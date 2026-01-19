import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'foreign' column to numeric (handle any potential parsing issues)
df['foreign'] = pd.to_numeric(df['foreign'], errors='coerce')

# Find the maximum and minimum values in the 'foreign' column
max_foreign = df['foreign'].max()
min_foreign = df['foreign'].min()

# Get the municipality names corresponding to max and min values
max_municipality = df.loc[df['foreign'].idxmax(), 'language']
min_municipality = df.loc[df['foreign'].idxmin(), 'language']

# Calculate the difference
difference = max_foreign - min_foreign

print(f"Final Answer: {max_municipality}, {difference:.1f}")