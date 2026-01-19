import pandas as pd

df = pd.read_csv('table.csv')
# Remove '%' sign and convert to float
df['% of national vote'] = df['% of national vote'].str.replace('%', '').astype(float)
# Calculate average percentage
avg_national_vote = df['% of national vote'].mean()
print(f"Final Answer: {avg_national_vote:.2f}")