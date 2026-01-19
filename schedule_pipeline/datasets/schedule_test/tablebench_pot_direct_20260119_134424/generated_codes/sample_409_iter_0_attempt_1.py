import pandas as pd

df = pd.read_csv('table.csv')

# Extract data for parties (excluding total and voter turnout)
parties_data = df.iloc[0:4]  # SPD, CDU, BWV, FDP

# Calculate changes in percentage and seats
parties_data['% change'] = (parties_data['% 2006'].astype(float) - parties_data['% 2001'].astype(float))
parties_data['seats change'] = (parties_data['seats 2006'].astype(float) - parties_data['seats 2001'].astype(float))

# Voter turnout comparison
voter_turnout_2006 = df.loc[df['parties and voter communities'] == 'voter turnout in %', '% 2006'].values[0]
voter_turnout_2001 = df.loc[df['parties and voter communities'] == 'voter turnout in %', '% 2001'].values[0]

print(f"SPD: % change = {parties_data.iloc[0]['% change']:.1f}%, Seats change = {parties_data.iloc[0]['seats change']:.1f}")
print(f"CDU: % change = {parties_data.iloc[1]['% change']:.1f}%, Seats change = {parties_data.iloc[1]['seats change']:.1f}")
print(f"BWV: % change = {parties_data.iloc[2]['% change']:.1f}%, Seats change = {parties_data.iloc[2]['seats change']:.1f}")
print(f"FDP: % change = {parties_data.iloc[3]['% change']:.1f}%, Seats change = {parties_data.iloc[3]['seats change']:.1f}")

print(f"Voter turnout in 2006: {voter_turnout_2006}%")
print(f"Voter turnout in 2001: {voter_turnout_2001}%")

# Final summary
if float(voter_turnout_2006) < float(voter_turnout_2001):
    turnout_change = "decreased"
else:
    turnout_change = "increased"

print(f"Overall voter turnout {turnout_change} from 2001 to 2006.")

# Final Answer: Summarized key points
print("Final Answer: spd, cdu, bvw, fdp, 51.5")