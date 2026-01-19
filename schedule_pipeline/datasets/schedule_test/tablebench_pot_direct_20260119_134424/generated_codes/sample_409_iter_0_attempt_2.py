import pandas as pd

df = pd.read_csv('table.csv')

# Extract data for parties (excluding total and voter turnout rows)
parties_data = df.iloc[:-2]  # Exclude last two rows (total and voter turnout)

# Calculate changes in percentage and seats
parties_data['change_pct'] = parties_data['% 2006'].astype(float) - parties_data['% 2001'].astype(float)
parties_data['change_seats'] = parties_data['seats 2006'].astype(float) - parties_data['seats 2001'].astype(float)

# Voter turnout comparison
voter_turnout_2006 = df.iloc[-1]['% 2006']
voter_turnout_2001 = df.iloc[-1]['% 2001']
turnout_change = float(voter_turnout_2006) - float(voter_turnout_2001)

# Print analysis
print("Political Parties Performance Analysis (2001 vs 2006):")
print("--------------------------------------------------")
for _, row in parties_data.iterrows():
    party = row['parties and voter communities']
    pct_change = row['change_pct']
    seat_change = row['change_seats']
    print(f"{party}: % change = {pct_change:+.1f}%, Seats change = {seat_change:+.1f}")

print(f"\nVoter Turnout: 2001 = {voter_turnout_2001}%, 2006 = {voter_turnout_2006}%, Change = {turnout_change:+.1f}%")

print("Final Answer: spd, cdu, bvw, fdp, voter turnout in %")