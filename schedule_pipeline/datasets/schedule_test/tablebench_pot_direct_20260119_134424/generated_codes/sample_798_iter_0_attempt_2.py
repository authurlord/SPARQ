import pandas as pd

df = pd.read_csv('table.csv')
# Filter data from 1947 to 1972
df_filtered = df[(df['Year'] >= '1947') & (df['Year'] <= '1972')]
# Convert Year to int for proper sorting
df_filtered['Year'] = df_filtered['Year'].astype(int)
df_filtered = df_filtered.sort_values('Year')
# Extract Conservative councillors
conservative_councillors = df_filtered['Conservative councillors'].astype(int)
# Calculate annual changes
annual_changes = conservative_councillors.diff().dropna()
# Calculate average annual change
avg_change = annual_changes.mean()
print(f"Final Answer: {avg_change:.2f}")