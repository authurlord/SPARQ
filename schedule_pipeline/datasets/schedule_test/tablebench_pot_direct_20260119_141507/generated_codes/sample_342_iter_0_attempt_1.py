import pandas as pd

df = pd.read_csv('table.csv')
# Filter countries with population > 40 million and count them
count_high_population = df[df['population'].astype(int) > 40000000].shape[0]
print(f"Final Answer: {count_high_population}")