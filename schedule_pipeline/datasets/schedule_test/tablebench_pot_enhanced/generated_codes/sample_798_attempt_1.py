import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years 1947 to 1972
df_filtered = df[(df['Year'] >= '1947') & (df['Year'] <= '1972')]
# Extract Conservative councillors values
conservative_councillors = df_filtered['Conservative councillors'].astype(int)
# Calculate average annual change
annual_change = (conservative_councillors.iloc[-1] - conservative_councillors.iloc[0]) / (len(conservative_councillors) - 1)
print(f"Final Answer: {annual_change:.2f}")