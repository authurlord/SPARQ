import pandas as pd

df = pd.read_csv('table.csv')
# Filter regions where 'e / vap ratio total' > 105%
filtered_df = df[df['e / vap ratio total'] > '105%']
# Convert ratio column to numeric for proper comparison
df['e / vap ratio total'] = df['e / vap ratio total'].str.rstrip('%').astype(float)
filtered_df = df[df['e / vap ratio total'] > 105]

# Sum enrolled men and women in filtered regions
total_enrolled = filtered_df['enrolled men'].sum() + filtered_df['enrolled women'].sum()
print(f"Final Answer: {total_enrolled}")