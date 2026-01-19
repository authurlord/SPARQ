import pandas as pd

df = pd.read_csv('table.csv')
# Convert all values to numeric (some entries like '1 630' need cleaning)
df['sunshine (hrs / year)'] = df['sunshine (hrs / year)'].str.replace(' ', '').astype(int)
df['rain (mm / year)'] = df['rain (mm / year)'].str.replace(' ', '').astype(int)
df['snow (days / year)'] = df['snow (days / year)'].astype(int)
df['storms (days / year)'] = df['storms (days / year)'].astype(int)
df['fog (days / year)'] = df['fog (days / year)'].astype(int)

# Compute correlation between snow and storms, and snow and fog
corr_storms = df['snow (days / year)'].corr(df['storms (days / year)'])
corr_fog = df['snow (days / year)'].corr(df['fog (days / year)'])

# Determine which has a greater absolute correlation
if abs(corr_storms) > abs(corr_fog):
    final_answer = "storms (days / year)"
else:
    final_answer = "fog (days / year)"

print(f"Final Answer: {final_answer}")