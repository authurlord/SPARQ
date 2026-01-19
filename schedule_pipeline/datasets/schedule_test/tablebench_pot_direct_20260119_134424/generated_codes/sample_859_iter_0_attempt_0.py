import pandas as pd

df = pd.read_csv('table.csv')
# Remove the 'Total:' row for accurate analysis
df = df[df['Party'] != 'Total:']
# Convert '%' column to float by removing the '%' sign and converting to float
df['%'] = df['%'].str.replace('%', '').astype(float)
# Find the party with the highest percentage
max_party = df.loc[df['%'].idxmax()]
# Get the percentage of Forward Bloc (Ruikar)
forward_bloc_ruikar_percent = df[df['Party'] == 'Forward Bloc (Ruikar)']['%'].values[0]
# Calculate the difference
difference = max_party['%'] - forward_bloc_ruikar_percent
# Print final answer in required format
print(f"Final Answer: {max_party['Party']}, {difference:.2f}")