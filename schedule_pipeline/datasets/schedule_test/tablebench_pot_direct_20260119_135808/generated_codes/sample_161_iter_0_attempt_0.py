import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'total viewers' to numeric, removing any non-numeric characters
df['total viewers'] = pd.to_numeric(df['total viewers'], errors='coerce')

# Convert 'share' to numeric, removing '%' and converting to float
df['share'] = df['share'].str.replace('%', '').astype(float)

# Convert 'bbc one weekly ranking' to numeric
df['bbc one weekly ranking'] = pd.to_numeric(df['bbc one weekly ranking'], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Calculate correlation coefficients
corr_total_viewers = df['total viewers'].corr(df['bbc one weekly ranking'])
corr_share = df['share'].corr(df['bbc one weekly ranking'])

# Check if any correlation is significant (e.g., |r| > 0.5)
if abs(corr_total_viewers) > 0.5:
    result = "total viewers"
elif abs(corr_share) > 0.5:
    result = "share"
else:
    result = "no clear impact"

print(f"Final Answer: {result}")