import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Drop the first row (header row) which is just labels
df = df.iloc[1:]

# Clean the data: convert non-numeric values to NaN and drop invalid rows
df['Miloš Zeman'] = df['Miloš Zeman'].replace('-', pd.NA)
df['Miloš Zeman_1'] = df['Miloš Zeman_1'].replace('-', pd.NA)

# Only use the first column for Miloš Zeman (since the second is missing)
df['Miloš Zeman'] = pd.to_numeric(df['Miloš Zeman'], errors='coerce')

# Calculate total votes for Miloš Zeman per round (deputies + senators)
df['total_zeman'] = df['Miloš Zeman'].fillna(0)

# Calculate total votes per round (deputies + senators)
df['total_votes'] = df['Václav Klaus'] + df['Václav Klaus_1'] + df['Jaroslava Moserová'] + df['Jaroslava Moserová_1'] + df['Miloš Zeman'] + df['Miloš Zeman_1']

# Only consider valid rounds with non-zero total votes
df = df.dropna(subset=['total_zeman', 'total_votes'])

# Calculate percentage
df['percentage'] = df['total_zeman'] / df['total_votes']

# Find the first round where percentage > 0.5
result_round = df[df['percentage'] > 0.5]['Round'].iloc[0] if not df[df['percentage'] > 0.5].empty else None

print(f"Final Answer: {result_round}")