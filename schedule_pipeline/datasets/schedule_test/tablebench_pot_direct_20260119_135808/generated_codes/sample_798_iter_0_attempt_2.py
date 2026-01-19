import pandas as pd

df = pd.read_csv('table.csv')
# Filter data from 1947 to 1972
df_filtered = df[df['Year'].astype(int) >= 1947]
df_filtered = df_filtered[df_filtered['Year'].astype(int) <= 1972]

# Convert 'Conservative councillors' to integer
conservative_councillors = df_filtered['Conservative councillors'].astype(int)

# Calculate annual changes
annual_changes = conservative_councillors.diff().dropna()

# Calculate average annual change
avg_change = annual_changes.mean()

print(f"Final Answer: {avg_change:.1f}")