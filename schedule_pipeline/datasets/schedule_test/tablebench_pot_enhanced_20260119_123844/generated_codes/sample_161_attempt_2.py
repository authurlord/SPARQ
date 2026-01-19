import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total viewers' to numeric (remove commas and convert to int)
df['total viewers'] = df['total viewers'].str.replace(',', '').astype(int)

# Convert 'share' to numeric (remove % and convert to float)
df['share'] = df['share'].str.replace('%', '').astype(float)

# Calculate correlation between 'total viewers' and 'bbc one weekly ranking'
corr_viewers = df['total viewers'].corr(df['bbc one weekly ranking'])

# Calculate correlation between 'share' and 'bbc one weekly ranking'
corr_share = df['share'].corr(df['bbc one weekly ranking'])

# Check if any correlation is significant (absolute value > 0.5)
if abs(corr_viewers) > 0.5 or abs(corr_share) > 0.5:
    if abs(corr_viewers) > abs(corr_share):
        print("Final Answer: total viewers")
    else:
        print("Final Answer: share")
else:
    print("Final Answer: no clear impact")