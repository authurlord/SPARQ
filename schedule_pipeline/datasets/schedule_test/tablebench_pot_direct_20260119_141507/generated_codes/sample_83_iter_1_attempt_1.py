import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where total medals is 6 and nation is not "total"
nation_with_6_medals = df[(df['total'] == 6) & (df['nation'] != 'total')]
# Extract the nation name
if not nation_with_6_medals.empty:
    final_nation = nation_with_6_medals.iloc[0]['nation']
else:
    final_nation = None

print(f"Final Answer: {final_nation}")