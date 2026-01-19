import pandas as pd

df = pd.read_csv('table.csv')

# Clean the data by extracting the main athlete name (before parentheses)
def extract_winner(winner):
    if pd.isna(winner):
        return None
    return winner.split(' (')[0].strip()

# Apply the function to both columns
df['senior pga champion'] = df['senior pga championship'].apply(extract_winner)
df['senior players champ'] = df['senior players championship'].apply(extract_winner)

# Find rows where both champions are not null and match
matching_rows = df[
    (df['senior pga champion'].notna()) & 
    (df['senior players champ'].notna()) & 
    (df['senior pga champion'] == df['senior players champ'])
]

# Get the year(s) of matching wins
if not matching_rows.empty:
    result_year = matching_rows['year'].iloc[0]
else:
    result_year = None

print(f"Final Answer: {result_year}")