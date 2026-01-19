import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where number of hurricanes is 3 or fewer
filtered_rows = df[df['number of hurricanes'] <= 3]
# Count the number of such years
count_years = len(filtered_rows)
print(f"Final Answer: {count_years}")