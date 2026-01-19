import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total viewers' to numeric
df['total viewers'] = pd.to_numeric(df['total viewers'], errors='coerce')

# Convert 'share' to numeric by removing '%' and converting
df['share'] = pd.to_numeric(df['share'].str.replace('%', ''), errors='coerce')

# Convert 'bbc one weekly ranking' to numeric
df['bbc one weekly ranking'] = pd.to_numeric(df['bbc one weekly ranking'], errors='coerce')

# Calculate correlation coefficients
correlation_viewers = df['total viewers'].corr(df['bbc one weekly ranking'])
correlation_share = df['share'].corr(df['bbc one weekly ranking'])

# Check for significant influence (absolute correlation > 0.7)
if abs(correlation_viewers) > 0.7 or abs(correlation_share) > 0.7:
    if abs(correlation_viewers) > abs(correlation_share):
        print("Final Answer: total viewers")
    else:
        print("Final Answer: share")
else:
    print("Final Answer: no clear impact")