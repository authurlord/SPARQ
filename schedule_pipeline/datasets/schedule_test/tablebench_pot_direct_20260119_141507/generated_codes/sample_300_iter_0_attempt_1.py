import pandas as pd

df = pd.read_csv('table.csv')

# Filter delegates from Metro Manila (including 'manila', 'metro manila', 'caloocan city', 'antipolo', etc.)
# But specifically, we look for "Metro Manila" or "manila" in hometown
df_metro = df[df['hometown'].str.contains('metro manila|manila|caloocan city|antipolo', case=False, na=False)]

# Further filter only those with a "winner" result
winners = df_metro[df_metro['result'].str.contains('winner', case=False, na=False)]

# Count the number of such delegates
count_winners_metro = len(winners)

print(f"Final Answer: {count_winners_metro}")