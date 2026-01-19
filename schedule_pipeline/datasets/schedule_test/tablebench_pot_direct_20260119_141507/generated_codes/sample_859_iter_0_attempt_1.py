import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the total row (last row with percentage "-")
df_filtered = df[df['%'] != '-']

# Convert percentage column to float
df_filtered['%'] = df_filtered['%'].str.replace('%', '').astype(float)

# Find the party with the highest percentage
max_percentage_row = df_filtered.loc[df_filtered['%'].idxmax()]
highest_percentage_party = max_percentage_row['Party']
highest_percentage = max_percentage_row['%']

# Get the percentage of Forward Bloc (Ruikar)
ruikar_percentage = df_filtered[df_filtered['Party'] == 'Forward Bloc (Ruikar)']['%'].values[0]

# Calculate the difference
difference = highest_percentage - ruikar_percentage

print(f"Final Answer: {highest_percentage_party}, {difference:.2f}")