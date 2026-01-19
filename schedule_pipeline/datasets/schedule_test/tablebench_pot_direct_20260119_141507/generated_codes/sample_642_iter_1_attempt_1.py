import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'percentage lost' column: remove '%' and convert to float
df['percentage lost'] = df['percentage lost'].str.replace('%', '').astype(float)

# Compute correlation between 'starting weight (kg)' and 'percentage lost'
correlation = df['starting weight (kg)'].corr(df['percentage lost'])

print(f"Final Answer: {correlation:.3f}")