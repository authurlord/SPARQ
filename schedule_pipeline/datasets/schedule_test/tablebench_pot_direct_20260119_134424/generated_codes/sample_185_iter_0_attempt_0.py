import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['area (km square)'] = pd.to_numeric(df['area (km square)'])
df['population'] = pd.to_numeric(df['population'])

# Calculate correlation between area and population
correlation = df['area (km square)'].corr(df['population'])

# Determine the relationship based on correlation
if correlation > 0:
    relationship = "positive"
elif correlation < 0:
    relationship = "negative"
else:
    relationship = "no clear impact"

print(f"Final Answer: {relationship}")