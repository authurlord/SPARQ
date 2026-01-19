import pandas as pd

df = pd.read_csv('table.csv')

# Define what constitutes a "winning" result
winning_results = ['winner', 'top 5 finalist', 'first runner - up', 'second runner - up', 'finalist']

# Filter delegates from Metro Manila (case-insensitive, including variations)
metro_manila_hometown = df['hometown'].str.contains('metro manila', case=False, na=False) | \
                        df['hometown'].str.contains('manila', case=False, na=False)

# Filter for those with a winning result
winning_delegates = df[metro_manila_hometown & df['result'].isin(winning_results)]

# Count the number of such delegates
count = len(winning_delegates)
print(f"Final Answer: {count}")