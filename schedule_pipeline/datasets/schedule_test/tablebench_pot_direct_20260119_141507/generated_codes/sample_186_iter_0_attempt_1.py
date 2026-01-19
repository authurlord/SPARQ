import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'of candidates nominated' to integer
df['of candidates nominated'] = df['of candidates nominated'].str.replace(',', '').astype(int)
# Extract the columns for analysis
candidates = df['of candidates nominated']
popular_vote = df['% of popular vote'].str.replace('%', '').astype(float)

# Calculate correlation between candidates and popular vote
correlation = candidates.corr(popular_vote)
print(f"Final Answer: {correlation:.2f}")