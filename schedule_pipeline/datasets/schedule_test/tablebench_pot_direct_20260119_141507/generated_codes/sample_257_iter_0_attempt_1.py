import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Date Built' to datetime
df['Date Built'] = pd.to_datetime(df['Date Built'], format='%B %Y', errors='coerce')

# Find the earliest and latest dates
earliest_date = df['Date Built'].min()
latest_date = df['Date Built'].max()

# Calculate the difference in months
diff_months = (latest_date - earliest_date).days / 30.44  # Approximate days per month

print(f"Final Answer: {diff_months:.1f}")