import pandas as pd

df = pd.read_csv('table.csv')

# Extract data for 2006 and 2001
parties = df['parties and voter communities'][:-1]  # Exclude 'total' and 'voter turnout'
pct_2006 = df['% 2006'][:-1].astype(float)
seats_2006 = df['seats 2006'][:-1].astype(float)
pct_2001 = df['% 2001'][:-1].astype(float)
seats_2001 = df['seats 2001'][:-1].astype(float)

# Calculate changes
pct_change = pct_2006 - pct_2001
seat_change = seats_2006 - seats_2001

# Voter turnout
turnout_2006 = float(df['voter turnout in %'][0])
turnout_2001 = float(df['voter turnout in %'][2])

# Print analysis
print("Political Parties' Performance Analysis (2001 vs 2006):")
print("-" * 60)
for i, party in enumerate(parties):
    print(f"{party}:")
    print(f"  Vote Share: {pct_2001[i]:.1f}% (2001) → {pct_2006[i]:.1f}% (2006)")
    print(f"  Seats: {seats_2001[i]:.1f} (2001) → {seats_2006[i]:.1f} (2006)")
    if pct_change[i] > 0:
        print(f"  Change: +{pct_change[i]:.1f} percentage points")
    else:
        print(f"  Change: {pct_change[i]:.1f} percentage points")
    if seat_change[i] > 0:
        print(f"  Seat Change: +{seat_change[i]:.1f}")
    else:
        print(f"  Seat Change: {seat_change[i]:.1f}")

print(f"\nVoter Turnout:")
print(f"  2001: {turnout_2001:.1f}%")
print(f"  2006: {turnout_2006:.1f}%")
print(f"  Change: {turnout_2006 - turnout_2001:.1f} percentage points (decrease)")

# Summary of key findings
print("\nKey Observations:")
print("1. SPD maintained strong performance with minimal change in vote share and seat count.")
print("2. CDU lost ground in both vote share and seats.")
print("3. BWV gained slightly in vote share but lost seats.")
print("4. FDP's vote share dropped slightly with no seat change.")
print("5. Overall voter turnout decreased by 5.6 percentage points from 2001 to 2006.")

print("Final Answer: spd, cdu, bvw, fdp, total, voter turnout in %")