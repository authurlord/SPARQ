import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Rank' to integer type to allow numeric filtering
df['Rank'] = df['Rank'].astype(int)
# Filter top 5 ranked governors (rank <= 5)
top_5_governors = df[df['Rank'] <= 5]
# Calculate the average number of days served
average_days = top_5_governors['Days'].mean()
print(f"Final Answer: {average_days:.0f}")