import pandas as pd

df = pd.read_csv('table.csv')

# Extract year from "Date Built" and "Date Withdrawn"
df['Date Built'] = df['Date Built'].str.extract(r'(\d{4})').astype(int)
df['Date Withdrawn'] = df['Date Withdrawn'].str.extract(r'(\d{4})').astype(int)

# Filter locomotives built in 1938
built_1938 = df[df['Date Built'] == 1938]

# Find the maximum withdrawal year among those built in 1938
max_withdrawal_year = built_1938['Date Withdrawn'].max()

# Calculate the maximum number of years in service
max_years_in_service = max_withdrawal_year - 1938

print(f"Final Answer: {max_years_in_service}")