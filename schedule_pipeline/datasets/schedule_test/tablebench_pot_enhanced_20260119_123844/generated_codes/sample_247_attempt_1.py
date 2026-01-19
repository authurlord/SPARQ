import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'weight lost (kg)' to numeric for sorting
df['weight lost (kg)'] = pd.to_numeric(df['weight lost (kg)'])
# Sort by weight lost in descending order
df_sorted = df.sort_values(by='weight lost (kg)', ascending=False)
# Select top 3 contestants
top_3 = df_sorted.head(3)
# Convert 'percentage lost' to numeric (remove % and convert to float)
top_3['percentage lost'] = top_3['percentage lost'].str.replace('%', '').astype(float)
# Calculate average percentage lost
avg_percentage_lost = top_3['percentage lost'].mean()
print(f"Final Answer: {avg_percentage_lost:.2f}")