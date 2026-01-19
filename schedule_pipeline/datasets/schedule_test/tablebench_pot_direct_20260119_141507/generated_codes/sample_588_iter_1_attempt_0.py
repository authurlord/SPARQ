import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert 'of national votes' to numeric
df['of national votes'] = df['of national votes'].str.replace('%', '').astype(float)

# Find the value for 1965
national_votes_1965 = df[df['election'] == '1965']['of national votes'].values[0]

# Increase by 10%
increased_votes = national_votes_1965 * 1.10

print(f"Final Answer: {increased_votes:.0f}")