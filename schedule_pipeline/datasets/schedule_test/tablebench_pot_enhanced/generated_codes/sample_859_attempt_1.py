import pandas as pd

df = pd.read_csv('table.csv')
# Convert '%' column to float by removing the '%' sign and converting to float
df['%'] = df['%'].str.replace('%', '').astype(float)

# Find the party with the highest percentage of votes
max_party = df.loc[df['%'].idxmax()]
highest_percentage = max_party['%']
party_name = max_party['Party']

# Find the percentage for Forward Bloc (Ruikar)
forward_bloc_ruikar = df[df['Party'] == 'Forward Bloc (Ruikar)']['%'].values[0]

# Calculate the difference
difference = highest_percentage - forward_bloc_ruikar

print(f"Final Answer: {party_name}, {difference:.2f}")