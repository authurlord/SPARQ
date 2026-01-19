import pandas as pd

df = pd.read_csv('table.csv')
# Filter regions with population > 4 million and count them
count_pop_over_4m = df[df['population'] > 4000000].shape[0]
print(f"Final Answer: {count_pop_over_4m}")