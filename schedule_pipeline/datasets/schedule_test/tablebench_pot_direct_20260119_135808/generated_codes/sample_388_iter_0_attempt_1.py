import pandas as pd

df = pd.read_csv('table.csv')
# Count rows where 'Source' contains 'Psalms'
psalm_count = df[df['Source'].str.contains('Psalms', na=False)].shape[0]
print(f"Final Answer: {psalm_count}")