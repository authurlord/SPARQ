import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Rank' to numeric, coercing errors to NaN if any
df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')

# Filter top 5 ranked governors (rank <= 5)
top_5_governors = df[df['Rank'] <= 5]

# Calculate average number of days served
average_days = top_5_governors['Days'].mean()

print(f"Final Answer: {average_days:.0f}")