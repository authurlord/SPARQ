import pandas as pd

df = pd.read_csv('table.csv')

# Extract relevant data
parties = df['parties and voter communities'][:-1]  # Exclude 'total' and 'voter turnout'
percent_2006 = df['% 2006'][:-1].astype(float)
seats_2006 = df['seats 2006'][:-1].astype(float)
percent_2001 = df['% 2001'][:-1].astype(float)
seats_2001 = df['seats 2001'][:-1].astype(float)

# Calculate changes
percent_change = percent_2006 - percent_2001
seat_change = seats_2006 - seats_2001

# Voter turnout
turnout_2006 = float(df['voter turnout in %'].iloc[0])
turnout_2001 = float(df['voter turnout in %'].iloc[1])

# Print analysis
print("Political Parties Performance Analysis (2001 vs 2006):")
print("--------------------------------------------------")
for i, party in enumerate(parties):
    print(f"{party}:")
    print(f"  Vote share: {percent_2006[i]:.1f}% (2006) vs {percent_2001[i]:.1f}% (2001)")
    print(f"  Seats: {seats_2006[i]:.1f} (2006) vs {seats_2001[i]:.1f} (2001)")
    print(f"  Change: {percent_change[i]:+.1f}% in vote share, {seat_change[i]:+.1f} seats")

print(f"\nVoter Turnout:")
print(f"  2006: {turnout_2006:.1f}%")
print(f"  2001: {turnout_2001:.1f}%")
print(f"  Change: {turnout_2006 - turnout_2001:+.1f}%")

print("\nFinal Answer: spd, cdu, bvw, fdp, total, voter turnout in %")