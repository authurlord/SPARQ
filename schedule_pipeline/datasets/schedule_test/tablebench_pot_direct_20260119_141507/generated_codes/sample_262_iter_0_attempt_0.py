import pandas as pd

df = pd.read_csv('table.csv')
# Filter regions where 'e / vap ratio total' > 105%
filtered_df = df[df['e / vap ratio total'] > '105%']
# Convert 'e / vap ratio total' to numeric for proper comparison (remove % and convert)
filtered_df['e / vap ratio total'] = filtered_df['e / vap ratio total'].str.replace('%', '').astype(float)
filtered_df = filtered_df[filtered_df['e / vap ratio total'] > 105]

# Sum enrolled men and women
total_enrolled = filtered_df['enrolled men'].sum() + filtered_df['enrolled women'].sum()
print(f"Final Answer: {total_enrolled}")