import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'share of votes' to numeric by removing '%' and converting to float
df['share of votes'] = df['share of votes'].str.replace('%', '').astype(float)

# Find the election with the highest and lowest share of votes
max_votes = df.loc[df['share of votes'].idxmax(), 'election']
min_votes = df.loc[df['share of votes'].idxmin(), 'election']
difference = df['share of votes'].max() - df['share of votes'].min()

print(f"Final Answer: {max_votes}, {min_votes}, {difference:.1f}")