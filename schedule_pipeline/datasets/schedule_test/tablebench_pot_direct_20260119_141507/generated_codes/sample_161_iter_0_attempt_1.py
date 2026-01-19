import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total viewers' and 'share' to numeric (handle % correctly)
df['total viewers'] = pd.to_numeric(df['total viewers'], errors='coerce')
df['share'] = pd.to_numeric(df['share'].str.replace('%', ''), errors='coerce')

# Drop rows with NaN due to conversion
df = df.dropna()

# Compute correlation with 'bbc one weekly ranking'
correlation_total_viewers = df['total viewers'].corr(df['bbc one weekly ranking'])
correlation_share = df['share'].corr(df['bbc one weekly ranking'])

# Check if either correlation is significant (absolute value > 0.3)
if abs(correlation_total_viewers) > 0.3 or abs(correlation_share) > 0.3:
    # Determine which factor(s) have significant influence
    if abs(correlation_total_viewers) > 0.3:
        significant_factors = 'total viewers'
    elif abs(correlation_share) > 0.3:
        significant_factors = 'share'
    else:
        significant_factors = 'no clear impact'
else:
    significant_factors = 'no clear impact'

print(f"Final Answer: {significant_factors}")