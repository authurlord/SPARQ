import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Death toll' to integer for proper sorting
df['Death toll'] = pd.to_numeric(df['Death toll'], errors='coerce')
# Sort by 'Death toll' in descending order
df_sorted = df.sort_values(by='Death toll', ascending=False)
# Select top 5 rows
top_5 = df_sorted.head(5)
# Calculate average magnitude
avg_magnitude = top_5['Magnitude'].mean()
print(f"Final Answer: {avg_magnitude:.1f}")