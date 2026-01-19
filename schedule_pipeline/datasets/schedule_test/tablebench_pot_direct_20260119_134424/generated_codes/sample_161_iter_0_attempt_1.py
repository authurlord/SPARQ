import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total viewers' and 'share' to numeric
df['total viewers'] = pd.to_numeric(df['total viewers'], errors='coerce')
df['share'] = df['share'].str.replace('%', '').astype(float)
df['bbc one weekly ranking'] = pd.to_numeric(df['bbc one weekly ranking'], errors='coerce')

# Calculate correlation coefficients
correlation_viewers = df['total viewers'].corr(df['bbc one weekly ranking'])
correlation_share = df['share'].corr(df['bbc one weekly ranking'])

# Check if any correlation is strong (|r| > 0.5)
if abs(correlation_viewers) > 0.5 or abs(correlation_share) > 0.5:
    if abs(correlation_viewers) > abs(correlation_share):
        influence = 'total viewers'
    else:
        influence = 'share'
else:
    influence = 'no clear impact'

print(f"Final Answer: {influence}")