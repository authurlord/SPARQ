import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'number of hurricanes' is 3 or fewer
count_years = df[df['number of hurricanes'] <= 3].shape[0]
print(f"Final Answer: {count_years}")