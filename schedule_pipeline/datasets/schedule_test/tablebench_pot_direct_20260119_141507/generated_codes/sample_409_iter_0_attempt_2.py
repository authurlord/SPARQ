import pandas as pd

df = pd.read_csv('table.csv')

# Convert percentage columns to float
df['% 2006'] = df['% 2006'].str.replace('%', '').astype(float)
df['% 2001'] = df['% 2001'].str.replace('%', '').astype(float)
df['seats 2006'] = df['seats 2006'].str.replace(',', '').astype(float)
df['seats 2001'] = df['seats 2001'].str.replace(',', '').astype(float)

# Voter turnout
turnout_2006 = df.loc[df['parties and voter communities'] == 'voter turnout in %', '% 2006'].values[0]
turnout_2001 = df.loc[df['parties and voter communities'] == 'voter turnout in %', '% 2001'].values[0]

# Analyze changes for each party
party_analysis = []
for index, row in df.iterrows():
    if row['parties and voter communities'] not in ['total', 'voter turnout in %']:
        change_percent = row['% 2006'] - row['% 2001']
        change_seats = row['seats 2006'] - row['seats 2001']
        party_analysis.append({
            'party': row['parties and voter communities'],
            'change_percent': change_percent,
            'change_seats': change_seats
        })

# Print detailed analysis
print("Detailed Analysis:")
print(f"Voter turnout in 2006: {turnout_2006}%, in 2001: {turnout_2001}% — a decrease of {abs(turnout_2001 - turnout_2006)} percentage points.")
print("Key observations:")
for item in party_analysis:
    if item['change_percent'] > 0:
        print(f"  {item['party']} gained {item['change_percent']}% in support and {item['change_seats']} seats.")
    elif item['change_percent'] < 0:
        print(f"  {item['party']} lost {abs(item['change_percent'])}% in support and {abs(item['change_seats'])} seats.")
    else:
        print(f"  {item['party']} maintained stable support and seat count.")

# Final Answer: Highlight key changes and turnout
Final Answer: SPD, CDU, BWV, FDP, voter turnout decrease