import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Date Built' contains '1938'
df_1938 = df[df['Date Built'].str.contains('1938', na=False)]

# Extract the year from 'Date Withdrawn' and convert to integer
df_1938['Withdrawn Year'] = df_1938['Date Withdrawn'].str.extract(r'(\d{4})').astype(int)

# Find the maximum withdrawal year
max_withdrawal_year = df_1938['Withdrawn Year'].max()

# Calculate maximum years in service
max_years_in_service = max_withdrawal_year - 1938

print(f"Final Answer: {max_years_in_service}")