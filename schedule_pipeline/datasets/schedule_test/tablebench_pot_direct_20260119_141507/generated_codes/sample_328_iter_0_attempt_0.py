import pandas as pd

df = pd.read_csv('table.csv')
# Filter films directed by Mostofa Sarwar Farooki and check notes for Academy Awards submission
filtered = df[(df['Director'] == 'Mostofa Sarwar Farooki') & 
              (df['Notes'].str.contains('submission to Academy Awards', case=False, na=False))]
count = len(filtered)
print(f"Final Answer: {count}")