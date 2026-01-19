import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Work is 'The Barbara Stanwyck Show' and Category contains 'Outstanding Performance by an Actress in a Series'
filtered_row = df[(df['Work'] == 'The Barbara Stanwyck Show') & (df['Category'].str.contains('Outstanding Performance', case=False))]

if not filtered_row.empty:
    year = filtered_row.iloc[0]['Year']
    print(f"Final Answer: {year}")
else:
    print("Final Answer: Not found")