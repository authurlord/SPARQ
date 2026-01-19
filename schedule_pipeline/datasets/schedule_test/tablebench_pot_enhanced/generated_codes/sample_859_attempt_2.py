import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'Total:' row for accurate analysis
df = df[df['Party'] != 'Total:']

# Convert '%' column to float by removing the '%' sign and converting to float
df['%'] = df['%'].str.replace('%', '').astype(float)

# Find the party with the highest percentage of votes
max_party = df.loc[df['%'].idxmax()]
highest_percentage = max_party['%']
forward_bloc_ruikar_percentage = df[df['Party'] == 'Forward Bloc (Ruikar)']['%'].values[0]
difference = highest_percentage - forward_bloc_ruikar_percentage

print(f"Final Answer: {max_party['Party']}, {difference:.2f}")