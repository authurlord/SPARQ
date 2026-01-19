import pandas as pd

df = pd.read_csv('table.csv')
# Filter data from 1947 to 1972
df_filtered = df[(df['Year'] >= '1947') & (df['Year'] <= '1972')]
# Convert Year to int for proper sorting
df_filtered['Year'] = df_filtered['Year'].astype(int)
# Sort by Year
df_filtered = df_filtered.sort_values('Year')
# Calculate annual change in Conservative councillors
annual_change = df_filtered['Conservative councillors'].diff().dropna()
# Calculate average annual change
avg_annual_change = annual_change.mean()
print(f"Final Answer: {avg_annual_change:.2f}")