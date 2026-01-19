import pandas as pd

df = pd.read_csv('table.csv')
# Filter for delegates from Metro Manila and those who won
metro_manila_winners = df[(df['hometown'].str.contains('metro manila', case=False, na=False)) & (df['result'] == 'winner')]
count_winners = len(metro_manila_winners)
print(f"Final Answer: {count_winners}")