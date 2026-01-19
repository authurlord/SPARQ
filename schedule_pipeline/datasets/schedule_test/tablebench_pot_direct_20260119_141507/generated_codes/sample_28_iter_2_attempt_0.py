import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Days' column to numeric by removing commas
df['Days'] = df['Days'].str.replace(',', '').astype(int)
# Filter top 5 ranked governors (Rank <= 5)
top_5_governors = df[df['Rank'] <= 5]
# Calculate average days served
average_days = top_5_governors['Days'].mean()
print(f"Final Answer: {average_days:.0f}")