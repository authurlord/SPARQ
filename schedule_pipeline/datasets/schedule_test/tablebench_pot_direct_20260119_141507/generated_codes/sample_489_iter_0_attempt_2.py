import pandas as pd

df = pd.read_csv('table.csv')

# Convert relevant columns to numeric, handling non-numeric values
df['Year Ended December 31, 2018 (In cents, except percentage changes)'] = pd.to_numeric(
    df['Year Ended December 31, 2018 (In cents, except percentage changes)'], errors='coerce'
)
df['Percent Increase (Decrease) (In cents, except percentage changes)'] = pd.to_numeric(
    df['Percent Increase (Decrease) (In cents, except percentage changes)'], errors='coerce'
)

# Identify rows with extreme values or anomalies
anomalies = []

# Check for extremely high values (e.g., > 100 in cents)
high_values = df[df['Year Ended December 31, 2018 (In cents, except percentage changes)'] > 100]
if not high_values.empty:
    anomalies.extend(high_values['Row Header'].tolist())

# Check for extremely high percentage changes (> 50%)
high_pct_changes = df[df['Percent Increase (Decrease) (In cents, except percentage changes)'] > 50]
if not high_pct_changes.empty:
    anomalies.extend(high_pct_changes['Row Header'].tolist())

# Check for values with '—' or 'nm (1)' or missing data
missing_or_special = df[
    df['Year Ended December 31, 2018 (In cents, except percentage changes)'].isna() |
    df['Row Header'].str.contains('—', case=False) |
    df['Row Header'].str.contains('nm', case=False)
]
if not missing_or_special.empty:
    anomalies.extend(missing_or_special['Row Header'].tolist())

# Remove duplicates and ensure clean list
anomalies = list(set(anomalies))

print(f"Final Answer: {', '.join(anomalies)}")