import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'Month' contains 'September'
september_hurricanes = df[df['Month'].str.contains('September', na=False)]
# Count the number of hurricanes in September
count_september = len(september_hurricanes)
print(f"Final Answer: {count_september}")