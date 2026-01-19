import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'of candidates nominated' to numeric and '% of popular vote' to float
df['of candidates nominated'] = pd.to_numeric(df['of candidates nominated'], errors='coerce')
df['% of popular vote'] = df['% of popular vote'].str.replace('%', '').astype(float)

# Calculate the correlation between the two variables
correlation = df['of candidates nominated'].corr(df['% of popular vote'])

print(f"Final Answer: {correlation:.2f}")