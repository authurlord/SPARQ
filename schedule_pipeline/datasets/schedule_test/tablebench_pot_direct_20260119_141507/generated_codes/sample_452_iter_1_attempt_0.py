import pandas as pd

df = pd.read_csv('table.csv')

# Convert revenue, profit, and market value to numeric (handle strings like '-16')
df['revenues (us billion)'] = pd.to_numeric(df['revenues (us billion)'], errors='coerce')
df['profit (us billion)'] = pd.to_numeric(df['profit (us billion)'].str.replace('-', '', regex=False), errors='coerce')
df['market value (us billion)'] = pd.to_numeric(df['market value (us billion)'], errors='coerce')

# Identify potential outliers
outliers = []

# Revenue outlier: extremely high (e.g., > 1000)
if df['revenues (us billion)'].max() > 1000:
    outliers.append("Hellenic Telecom has unusually high revenue (1000 billion USD)")

# Profit outlier: negative or very low
if df['profit (us billion)'].min() < -10:
    outliers.append("Piraeus Bank has a very negative profit (-8.6 billion USD)")

# Market value outlier: extremely high
if df['market value (us billion)'].max() > 100:
    outliers.append("OPAP has an unusually high market value (100 billion USD)")

# Check if any company has negative profit with high assets (possible distress)
distressed_banks = df[(df['profit (us billion)'] < 0) & (df['assets (us billion)'] > 50)]
if not distressed_banks.empty:
    distressed_names = ', '.join(distressed_banks['company'])
    outliers.append(f"Distressed banks with negative profit: {distressed_names}")

# Print identified unusual patterns
print(", ".join(outliers))