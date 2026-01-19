import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'podiums' to numeric, coercing errors to NaN
df['podiums'] = pd.to_numeric(df['podiums'], errors='coerce')
# Remove rows with NaN values (invalid podium counts)
df_clean = df.dropna(subset=['podiums'])
# Calculate the average number of podiums
average_podiums = df_clean['podiums'].mean()
print(f"Final Answer: {average_podiums:.1f}")