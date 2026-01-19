import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total viewers' to numeric (remove commas and convert to int)
df['total viewers'] = df['total viewers'].str.replace(',', '').astype(int)

# Convert 'share' to numeric (remove % and convert to float)
df['share'] = df['share'].str.replace('%', '').astype(float)

# Convert 'bbc one weekly ranking' to numeric
df['bbc one weekly ranking'] = df['bbc one weekly ranking'].astype(int)

# Calculate correlation coefficients
correlation_viewers = df['total viewers'].corr(df['bbc one weekly ranking'])
correlation_share = df['share'].corr(df['bbc one weekly ranking'])

# Check if any correlation is significant (absolute value > 0.7)
if abs(correlation_viewers) > 0.7 or abs(correlation_share) > 0.7:
    if abs(correlation_viewers) > abs(correlation_share):
        print("Final Answer: total viewers")
    else:
        print("Final Answer: share")
else:
    print("Final Answer: no clear impact")