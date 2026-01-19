import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'seats' to numeric, coercing errors to NaN if any
df['seats'] = pd.to_numeric(df['seats'], errors='coerce')
# Filter rows where seats >= 75 and count them
elections_with_75_plus_seats = df[df['seats'] >= 75].shape[0]
print(f"Final Answer: {elections_with_75_plus_seats}")