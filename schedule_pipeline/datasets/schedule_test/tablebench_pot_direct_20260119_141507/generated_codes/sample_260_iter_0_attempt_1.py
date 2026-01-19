import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where Date Built contains '1938'
df_1938 = df[df['Date Built'].str.contains('1938')]

# Extract the year from 'Date Withdrawn' (assuming format like "December 1964")
df_1938['withdrawal_year'] = df_1938['Date Withdrawn'].str.extract(r'(\d{4})')[0].astype(int)

# Find the maximum withdrawal year
max_withdrawal_year = df_1938['withdrawal_year'].max()

# Calculate the maximum service duration in years
max_years_in_service = max_withdrawal_year - 1938

print(f"Final Answer: {max_years_in_service}")