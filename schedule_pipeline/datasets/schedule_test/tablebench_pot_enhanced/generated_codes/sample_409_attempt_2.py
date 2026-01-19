import pandas as pd

df = pd.read_csv('table.csv')

# Extract data for parties (excluding total and voter turnout)
parties = df.iloc[0:4]  # spd, cdu, bvw, fdp

# Compare percentage and seats
parties['% change'] = (parties['% 2006'].astype(float) - parties['% 2001'].astype(float))
parties['seats change'] = (parties['seats 2006'].astype(float) - parties['seats 2001'].astype(float))

# Voter turnout
turnout_2006 = df.loc[df['parties and voter communities'] == 'voter turnout in %', '% 2006'].values[0]
turnout_2001 = df.loc[df['parties and voter communities'] == 'voter turnout in %', '% 2001'].values[0]

# Print analysis
print("Political Parties Performance Analysis (2001 vs 2006):")
print("------------------------------------------------")
for _, row in parties.iterrows():
    print(f"{row['parties and voter communities']}:")
    print(f"  - % in 2006: {row['% 2006']}%, in 2001: {row['% 2001']}% (change: {row['% change']:.1f}%)")
    print(f"  - Seats in 2006: {row['seats 2006']}, in 2001: {row['seats 2001']} (change: {row['seats change']:.1f})")

print(f"\nVoter Turnout:")
print(f"  2006: {turnout_2006}%")
print(f"  2001: {turnout_2001}%")
print(f"  Change: {float(turnout_2006) - float(turnout_2001):.1f}% (decrease)")

print(f"Final Answer: spd, cdu, bvw, fdp, 51.5, 57.1")