import pandas as pd

df = pd.read_csv('table.csv')
# Convert '2001 - 05' column to float and filter values > 0.3
count = (df['2001 - 05'].astype(float) > 0.3).sum()
print(f"Final Answer: {count}")