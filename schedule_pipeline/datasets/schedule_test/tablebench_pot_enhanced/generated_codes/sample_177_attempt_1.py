import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to numeric
df['annual ridership (2012)'] = pd.to_numeric(df['annual ridership (2012)'])
df['lines'] = pd.to_numeric(df['lines'])

# Calculate correlation
correlation = df['lines'].corr(df['annual ridership (2012)'])

# Determine impact based on correlation
if correlation > 0:
    impact = "positive"
elif correlation < 0:
    impact = "negative"
else:
    impact = "no clear impact"

print(f"Final Answer: {impact}")