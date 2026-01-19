import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where percentage increase is greater than 5%
increased_categories = df[df['Percent Increase (Decrease) (In cents, except percentage changes)'] > 5]
# Count the number of such categories
count = len(increased_categories)
print(f"Final Answer: {count}")