import pandas as pd

df = pd.read_csv('table.csv')

# Filter for African Championships and 20 km walk
african_championships_20km = df[(df['Competition'] == 'African Championships') & (df['Event'] == '20 km walk')]

# Convert time to a comparable format (remove 'CR' or '?' if present)
african_championships_20km = african_championships_20km[african_championships_20km['Notes'].str.contains('DNF|DQ|–|?') == False]

# Sort by time (assuming format mm:ss or hh:mm:ss, and we can sort lexicographically due to fixed format)
african_championships_20km['Notes'] = african_championships_20km['Notes'].str.replace(' \(CR\)', '')
african_championships_20km = african_championships_20km.sort_values(by='Notes')

# Get the year of the fastest time
best_year = african_championships_20km.iloc[0]['Year']
print(f"Final Answer: {best_year}")