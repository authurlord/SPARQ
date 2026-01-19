import pandas as pd

df = pd.read_csv('table.csv')

# Check for missing values
missing_mintage_proof = df['mintage (proof)'].isna().sum()
missing_issue_price_proof = df['issue price (proof)'].isna().sum()

# Identify high issue price outlier
high_issue_price = df[df['issue price (proof)'] > 100]

# Identify very low mintage
low_mintage = df[df['mintage (proof)'] < 10000]

# Identify entries with "no bu exists"
no_bu_exists = df[df['mintage (bu)'].str.contains('no bu exists', case=False) | df['issue price (bu)'].str.contains('n / a', case=False)]

print(f"Missing values in mintage (proof): {missing_mintage_proof}")
print(f"Missing values in issue price (proof): {missing_issue_price_proof}")
print(f"High issue price (>$100): {high_issue_price[['year', 'theme', 'artist', 'issue price (proof)']]}")
print(f"Very low mintage (<10,000): {low_mintage[['year', 'theme', 'artist', 'mintage (proof)']]}")
print(f"Entries with 'no bu exists' or 'n / a': {no_bu_exists[['year', 'theme', 'artist']]}")

Final Answer: missing values, high issue price, low mintage, inconsistent BU data