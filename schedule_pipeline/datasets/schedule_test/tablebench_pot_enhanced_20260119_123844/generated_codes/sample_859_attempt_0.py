import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'Total:' row for accurate analysis
df = df[df['Party'] != 'Total:']

# Convert '%' column to float by removing the '%' sign and converting to float
df['%'] = df['%'].str.replace('%', '').astype(float)

# Find the party with the highest percentage of votes
max_vote_party = df.loc[df['%'].idxmax()]
highest_percentage = max_vote_party['%']
forward_bloc_ruikar_percentage = df[df['Party'] == 'Forward Bloc (Ruikar)']['%'].values[0]

# Calculate the difference
difference = highest_percentage - forward_bloc_ruikar_percentage

# Output the result
print(f"Final Answer: Indian National Congress, {difference:.2f}")