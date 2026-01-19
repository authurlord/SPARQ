import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where percentage increase is greater than 5%
increased_categories = df[df['Percent Increase (Decrease) (In cents, except percentage changes)'].astype(float) > 5]
count = increased_categories.shape[0]
print(f"Final Answer: {count}")