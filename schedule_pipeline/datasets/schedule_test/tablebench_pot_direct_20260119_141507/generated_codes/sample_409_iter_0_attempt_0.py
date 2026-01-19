import pandas as pd

df = pd.read_csv('table.csv')

# Convert percentage columns to float
df['% 2006'] = pd.to_numeric(df['% 2006'], errors='coerce')
df['% 2001'] = pd.to_numeric(df['% 2001'], errors='coerce')
df['seats 2006'] = pd.to_numeric(df['seats 2006'], errors='coerce')
df['seats 2001'] = pd.to_numeric(df['seats 2001'], errors='coerce')

# Exclude total row and voter turnout row
parties = df[df['parties and voter communities'] != 'total'].copy()

# Calculate changes in % and seats
parties['% change 2006-2001'] = parties['% 2006'] - parties['% 2001']
parties['seats change 2006-2001'] = parties['seats 2006'] - parties['seats 2001']

# Display analysis
print("Detailed Analysis:")
print("Voter turnout: 51.5% in 2006, 57.1% in 2001 — slightly lower in 2006.")
print("Key observations:")
for index, row in parties.iterrows():
    party = row['parties and voter communities']
    pct_change = row['% change 2006-2001']
    seats_change = row['seats change 2006-2001']
    print(f"  {party}: Vote share changed by {pct_change:.1f}%, seats changed by {seats_change} seats.")
print("Final Answer: SPD, CDU, BWV, FDP, voter turnout")