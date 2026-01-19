import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where 'e / vap ratio total' > 105%
filtered_df = df[df['e / vap ratio total'] > '105%']

# Convert enrolled men and women to integers
filtered_df['enrolled men'] = filtered_df['enrolled men'].astype(int)
filtered_df['enrolled women'] = filtered_df['enrolled women'].astype(int)

# Calculate total enrolled men and women
total_enrolled_men = filtered_df['enrolled men'].sum()
total_enrolled_women = filtered_df['enrolled women'].sum()

# Final total
total_enrolled = total_enrolled_men + total_enrolled_women

print(f"Final Answer: {total_enrolled}")