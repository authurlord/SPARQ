import pandas as pd

df = pd.read_csv('table.csv')
# Count jurisdictions where percent for is greater than 70%
count_high_for = df[df['percent for'] > 70.0]['percent for'].count()
print(f"Final Answer: {count_high_for}")