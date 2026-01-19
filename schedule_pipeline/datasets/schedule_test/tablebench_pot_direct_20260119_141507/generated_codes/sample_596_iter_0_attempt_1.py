import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean the 'profit (us billion)' column by removing spaces and converting to float
df['profit (us billion)'] = df['profit (us billion)'].str.replace(' - ', '-').astype(float)

# Sort by 'revenues (us billion)' in descending order and take top 5
top_5 = df.sort_values(by='revenues (us billion)', ascending=False).head(5)

# Calculate the average profit of the top 5
avg_profit = top_5['profit (us billion)'].mean()

print(f"Final Answer: {avg_profit:.1f}")