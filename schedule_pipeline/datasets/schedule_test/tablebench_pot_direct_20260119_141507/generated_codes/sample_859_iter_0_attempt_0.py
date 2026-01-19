import pandas as pd

df = pd.read_csv('table.csv')
# Remove the total row (last row with 'Total:')
df = df[df['Party'] != 'Total:']

# Convert the '%' column to float for numerical comparison
df['%'] = df['%'].str.rstrip('%').astype(float)

# Find the party with the highest percentage
max_percentage_row = df.loc[df['%'].idxmax()]
highest_percentage_party = max_percentage_row['Party']
highest_percentage = max_percentage_row['%']

# Forward Bloc (Ruikar)'s percentage
forward_bloc_ruikar_percentage = 1.45

# Calculate the difference
difference = highest_percentage - forward_bloc_ruikar_percentage

print(f"Final Answer: {highest_percentage_party}, {difference:.2f}")