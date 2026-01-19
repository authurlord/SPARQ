import pandas as pd

df = pd.read_csv('table.csv')
# Convert the first column to integer for proper filtering
df['-'] = pd.to_numeric(df['-'])
# Filter data from 2012 to 2014
filtered_df = df[(df['-'] >= 2012) & (df['-'] <= 2014)]
# Sum the 'Films' column
total_films = filtered_df['Films'].sum()
print(f"Final Answer: {total_films}")