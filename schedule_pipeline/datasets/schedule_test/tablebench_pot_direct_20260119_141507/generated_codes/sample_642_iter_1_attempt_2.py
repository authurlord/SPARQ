import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'percentage lost' column by removing '%' and converting to float
df['percentage lost'] = df['percentage lost'].str.replace('%', '').astype(float)

# Calculate the correlation between 'starting weight' and 'percentage lost'
correlation = df['starting weight'].corr(df['percentage lost'])

print(f"Final Answer: {correlation:.2f}")