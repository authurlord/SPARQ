import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'podiums' column to numeric, coercing errors to NaN
df['podiums'] = pd.to_numeric(df['podiums'], errors='coerce')
# Calculate the average of podiums
avg_podiums = df['podiums'].mean()
print(f"Final Answer: {avg_podiums:.1f}")