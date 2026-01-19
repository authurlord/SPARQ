import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total viewers' and 'share' to numeric
df['total viewers'] = pd.to_numeric(df['total viewers'])
df['share'] = df['share'].str.rstrip('%').astype(float)
df['bbc one weekly ranking'] = pd.to_numeric(df['bbc one weekly ranking'])

# Calculate correlation
correlation_viewers = df['total viewers'].corr(df['bbc one weekly ranking'])
correlation_share = df['share'].corr(df['bbc one weekly ranking'])

# Check if any correlation is strong
if abs(correlation_viewers) > 0.7 or abs(correlation_share) > 0.7:
    if abs(correlation_viewers) > abs(correlation_share):
        print("Final Answer: total viewers")
    else:
        print("Final Answer: share")
else:
    print("Final Answer: no clear impact")