import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Date Built' to datetime
df['Date Built'] = pd.to_datetime(df['Date Built'], format='%B %Y')

# Find the earliest and latest 'Date Built'
earliest = df['Date Built'].min()
latest = df['Date Built'].max()

# Calculate the difference in months
diff_months = (latest.year - earliest.year) * 12 + (latest.month - earliest.month)
print(f"Final Answer: {diff_months}")