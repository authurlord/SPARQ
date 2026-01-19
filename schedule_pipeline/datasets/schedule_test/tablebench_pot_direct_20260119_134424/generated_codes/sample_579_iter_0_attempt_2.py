import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for years between 2000 and 2007
filtered_df = df[df['year'].astype(int).between(2000, 2007)]
# Calculate average quantity
avg_quantity = filtered_df['quantity'].astype(int).mean()
print(f"Final Answer: {avg_quantity:.1f}")