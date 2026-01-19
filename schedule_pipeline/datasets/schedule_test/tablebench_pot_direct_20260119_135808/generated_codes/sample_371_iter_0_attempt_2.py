import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where builder is 'brighton' and whenbuilt is '1948'
brighton_1948 = df[(df['builder'] == 'brighton') & (df['whenbuilt'] == '1948')]
count = len(brighton_1948)
print(f"Final Answer: {count}")