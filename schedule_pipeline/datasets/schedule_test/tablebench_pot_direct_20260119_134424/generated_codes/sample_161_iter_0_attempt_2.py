import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total viewers' to numeric
df['total viewers'] = pd.to_numeric(df['total viewers'])

# Convert 'share' to numeric by removing '%' and converting
df['share'] = pd.to_numeric(df['share'].str.replace('%', ''))

# Calculate correlation with 'bbc one weekly ranking'
correlation_viewers = df['total viewers'].corr(df['bbc one weekly ranking'])
correlation_share = df['share'].corr(df['bbc one weekly ranking'])

# Print correlation values for interpretation
print(f"Correlation with total viewers: {correlation_viewers:.3f}")
print(f"Correlation with share: {correlation_share:.3f}")

# Based on low correlation values, conclude
if abs(correlation_viewers) < 0.5 and abs(correlation_share) < 0.5:
    print("Final Answer: no clear impact")
else:
    # If any correlation is strong, we would report the influencing factor
    if abs(correlation_viewers) >= 0.5:
        print("Final Answer: total viewers")
    elif abs(correlation_share) >= 0.5:
        print("Final Answer: share")
    else:
        print("Final Answer: no clear impact")