import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the numeric value from 'of national votes' column (remove % and convert to float)
def parse_votes(vote_str):
    return float(vote_str.strip('%'))

# Apply the parsing function to the 'of national votes' column
df['of national votes'] = df['of national votes'].apply(parse_votes)

# Get the national votes for 1965
votes_1965 = df[df['election'] == '1965']['of national votes'].values[0]

# Increase by 10%
increased_votes = votes_1965 * 1.10

print(f"Final Answer: {increased_votes:.0f}")