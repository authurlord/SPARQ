import pandas as pd

df = pd.read_csv('table.csv')
# Exclude the 'total' row and filter years from 2000 to 2010
indian_admitted = df[df['year'].str.startswith('200')]['indians admitted'].dropna()
# Calculate the average
average_indians = indian_admitted.mean()
print(f"Final Answer: {average_indians:.0f}")