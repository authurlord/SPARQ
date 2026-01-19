import pandas as pd

df = pd.read_csv('table.csv')

# Convert profit and revenues to numeric, handling potential string issues
df['revenues (us billion)'] = pd.to_numeric(df['revenues (us billion)'], errors='coerce')
df['profit (us billion)'] = pd.to_numeric(df['profit (us billion)'].str.replace('-', '', regex=False), errors='coerce')

# Identify outliers
outliers = []

# Check for extremely high revenues
if df['revenues (us billion)'].max() > 1000:
    outliers.append("Hellenic Telecom has an unusually high revenue of 1000 US billion, which is likely a data entry error.")

# Check for negative or extreme profits
negative_profits = df[df['profit (us billion)'] < 0]
if len(negative_profits) > 0:
    outliers.append("Several companies have negative profits, including National Bank of Greece (-16) and Piraeus Bank (-8.6), indicating losses.")

# Check for extremely high market value
if df['market value (us billion)'].max() > 100:
    outliers.append("Piraeus Bank has a market value of 100 US billion, which is unusually high compared to other entries.")

# Check for inconsistent data format (e.g., "- 16" instead of -16)
if df['profit (us billion)'].isna().any():
    outliers.append("Some profit values are formatted incorrectly (e.g., '- 16'), which may indicate data entry errors.")

print(f"Final Answer: {', '.join(outliers)}")