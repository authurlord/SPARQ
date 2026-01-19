import pandas as pd

df = pd.read_csv('table.csv')

# Identify anomalies in mintage and issue price
anomalies = []

# Check for missing values
missing_mintage = df[df['mintage (proof)'].isna() | (df['mintage (proof)'] == 'n / a') | (df['mintage (proof)'] == 'no bu exists')]
missing_issue_price = df[df['issue price (proof)'].isna() | (df['issue price (proof)'] == 'n / a')]

# Check for extremely low mintage with high issue price
high_price_low_mintage = df[(df['mintage (proof)'].astype(str) != 'n / a') & 
                            (df['mintage (proof)'].astype(float) < 1000) & 
                            (df['issue price (proof)'].astype(float) > 50)]

# Check for inconsistency: mintage exists but issue price is missing
inconsistent_data = df[df['mintage (proof)'].notna() & df['issue price (proof)'].isna()]

# Compile anomalies
anomalies = [
    f"Missing mintage: {len(missing_mintage)} records",
    f"Missing issue price: {len(missing_issue_price)} records",
    f"High issue price with low mintage: {len(high_price_low_mintage)} records",
    f"Inconsistent data (mintage exists, issue price missing): {len(inconsistent_data)} records"
]

print(f"Final Answer: {', '.join(anomalies)}")