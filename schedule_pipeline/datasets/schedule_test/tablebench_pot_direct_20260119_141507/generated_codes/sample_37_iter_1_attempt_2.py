import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'average' column to numeric, coercing errors to NaN if any (though none expected)
df['average'] = pd.to_numeric(df['average'], errors='coerce')
# Sum the average ratings
total_average = df['average'].sum()
print(f"Final Answer: {total_average}")