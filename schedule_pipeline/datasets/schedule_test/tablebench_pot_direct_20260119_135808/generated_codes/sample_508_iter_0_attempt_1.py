import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'percent for' to float and find the jurisdiction with the highest value
max_support_jurisdiction = df.loc[df['percent for'].astype(float).idxmax(), 'jurisdiction']
print(f"Final Answer: {max_support_jurisdiction}")