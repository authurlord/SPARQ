import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Date Built' to datetime
df['Date Built'] = pd.to_datetime(df['Date Built'], format='%B %Y')

# Find earliest and latest 'Date Built'
earliest_date = df['Date Built'].min()
latest_date = df['Date Built'].max()

# Calculate difference in months
diff_months = (latest_date.year - earliest_date.year) * 12 + (latest_date.month - earliest_date.month)
print(f"Final Answer: {diff_months}")