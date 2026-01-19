import pandas as pd

df = pd.read_csv('table.csv')

# Function to extract the athlete name (before parentheses)
def extract_winner(winner):
    if pd.isna(winner):
        return ""
    return winner.split(' (')[0].strip()

# Apply the function to both columns
df['senior pga champion'] = df['senior pga championship'].apply(extract_winner)
df['senior players champ'] = df['senior players championship'].apply(extract_winner)

# Find rows where the same athlete won both titles
matched_rows = df[df['senior pga champion'] == df['senior players champ']]

# Extract the year
if not matched_rows.empty:
    year = matched_rows['year'].iloc[0]
    print(f"Final Answer: {year}")
else:
    print("Final Answer: None")