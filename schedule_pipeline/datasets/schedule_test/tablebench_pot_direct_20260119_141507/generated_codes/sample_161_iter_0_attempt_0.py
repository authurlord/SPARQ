import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total viewers' and 'share' to numeric (handle % properly)
df['total viewers'] = pd.to_numeric(df['total viewers'], errors='coerce')
df['share'] = df['share'].str.replace('%', '').astype(float) / 100.0

# Drop rows with NaN due to conversion
df = df.dropna()

# Calculate correlation with 'bbc one weekly ranking'
correlation_total_viewers = df['total viewers'].corr(df['bbc one weekly ranking'])
correlation_share = df['share'].corr(df['bbc one weekly ranking'])

# Check if either has a significant correlation (absolute value > 0.3)
if abs(correlation_total_viewers) > 0.3 or abs(correlation_share) > 0.3:
    # Determine which factor(s) have significant influence
    if abs(correlation_total_viewers) > 0.3:
        significant_factor = 'total viewers'
    elif abs(correlation_share) > 0.3:
        significant_factor = 'share'
    else:
        significant_factor = 'no clear impact'
else:
    significant_factor = 'no clear impact'

print(f"Final Answer: {significant_factor}")