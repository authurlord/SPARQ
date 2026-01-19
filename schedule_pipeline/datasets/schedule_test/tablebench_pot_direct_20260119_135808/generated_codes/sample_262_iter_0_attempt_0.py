import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'e / vap ratio total' to float by removing the % sign
df['e / vap ratio total'] = df['e / vap ratio total'].str.replace('%', '').astype(float)
# Filter rows where 'e / vap ratio total' > 105
filtered_df = df[df['e / vap ratio total'] > 105]
# Sum enrolled men and women
total_enrolled_men = filtered_df['enrolled men'].sum()
total_enrolled_women = filtered_df['enrolled women'].sum()
print(f"Final Answer: {total_enrolled_men}, {total_enrolled_women}")