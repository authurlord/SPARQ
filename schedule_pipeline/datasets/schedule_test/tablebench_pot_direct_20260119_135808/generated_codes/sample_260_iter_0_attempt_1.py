import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where the locomotive was built in 1938
df_1938 = df[df['Date Built'].str.contains('1938')]

# Convert Date Built and Date Withdrawn to year only
df_1938['Year Built'] = df_1938['Date Built'].str.extract(r'(\d{4})').astype(int)
df_1938['Year Withdrawn'] = df_1938['Date Withdrawn'].str.extract(r'(\d{4})').astype(int)

# Calculate the number of years in service
df_1938['Years in Service'] = df_1938['Year Withdrawn'] - df_1938['Year Built']

# Find the maximum number of years
max_years = df_1938['Years in Service'].max()

print(f"Final Answer: {max_years}")