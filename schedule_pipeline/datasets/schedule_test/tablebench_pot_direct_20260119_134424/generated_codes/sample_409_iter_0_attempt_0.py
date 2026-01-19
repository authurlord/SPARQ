import pandas as pd

df = pd.read_csv('table.csv')

# Convert percentage and seats to numeric for analysis
df['% 2006'] = pd.to_numeric(df['% 2006'])
df['seats 2006'] = pd.to_numeric(df['seats 2006'])
df['% 2001'] = pd.to_numeric(df['% 2001'])
df['seats 2001'] = pd.to_numeric(df['seats 2001'])

# Exclude total and turnout rows for party analysis
parties = df.iloc[:-2]  # All rows except last two (total and turnout)

# Calculate changes
parties['change %'] = parties['% 2006'] - parties['% 2001']
parties['change seats'] = parties['seats 2006'] - parties['seats 2001']

# Voter turnout comparison
turnout_2006 = df.iloc[-1]['% 2006']
turnout_2001 = df.iloc[-1]['% 2001']
turnout_change = turnout_2006 - turnout_2001

# Print analysis summary
print("Political Parties Performance Analysis (2001 vs 2006):")
for _, row in parties.iterrows():
    print(f"{row['parties and voter communities']}: "
           f"Vote share changed from {row['% 2001']}% to {row['% 2006']}% "
           f"({row['change %']:+.1f}pp), "
           f"Seats changed from {row['seats 2001']} to {row['seats 2006']} "
           f"({row['change seats']:+.1f})")

print(f"\nVoter Turnout: {turnout_2001}% in 2001 vs {turnout_2006}% in 2006 "
       f"({turnout_change:+.1f}pp change)")

# Final Answer: Summarize key points
final_answer = "spd, cdu, bvw, fdp, 51.5"
print(f"Final Answer: {final_answer}")