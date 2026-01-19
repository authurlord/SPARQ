import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'civil liberties' to numeric (handles string values)
df['civil liberties'] = pd.to_numeric(df['civil liberties'], errors='coerce')

# Get civil liberties scores for 1972 and 1980
civil_1972 = df[df['year'] == '1972']['civil liberties'].iloc[0]
civil_1980 = df[df['year'] == '1980']['civil liberties'].iloc[0]

# Calculate percentage change
percentage_change = ((civil_1980 - civil_1972) / civil_1972) * 100

print(f"Final Answer: {percentage_change:.1f}")