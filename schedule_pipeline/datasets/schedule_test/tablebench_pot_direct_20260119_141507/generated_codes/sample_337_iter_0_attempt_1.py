import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Event is '200 m' and Notes contains 'PB' or '1st PB'
pb_rows = df[(df['Event'] == '200 m') & (df['Notes'].str.contains('PB', case=False) | df['Notes'].str.contains('1st PB', case=False))]
# Extract the year if such a row exists
if not pb_rows.empty:
    year = pb_rows.iloc[0]['Year']
    print(f"Final Answer: {year}")
else:
    print("Final Answer: None")