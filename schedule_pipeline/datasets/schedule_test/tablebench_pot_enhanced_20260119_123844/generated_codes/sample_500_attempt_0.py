import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Death toll' to integer for proper sorting
df['Death toll'] = pd.to_numeric(df['Death toll'], errors='coerce')
# Sort by 'Death toll' in descending order and take top 5
top_5 = df.nlargest(5, 'Death toll')
# Calculate average magnitude
avg_magnitude = top_5['Magnitude'].mean()
print(f"Final Answer: {avg_magnitude:.1f}")