import pandas as pd

df = pd.read_csv('table.csv')
# Convert crowd to integer for comparison
df['crowd'] = pd.to_numeric(df['crowd'], errors='coerce')
# Filter rows where venue is 'waverley park' and crowd > 40000
filtered_df = df[(df['venue'] == 'waverley park') & (df['crowd'] > 40000)]
# Find the row with the highest margin
max_margin_row = filtered_df.loc[filtered_df['margin'].idxmax()]
# Extract the winning team
winner = max_margin_row['winners']
print(f"Final Answer: {winner}")