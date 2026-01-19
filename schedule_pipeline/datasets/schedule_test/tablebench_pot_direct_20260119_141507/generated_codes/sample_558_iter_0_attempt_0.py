import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for years from 1935 to 1943 inclusive
filtered_df = df[(df['Year'].astype(str).str.startswith('1935') | 
                  df['Year'].astype(str).str.startswith('1936') | 
                  df['Year'].astype(str).str.startswith('1937') | 
                  df['Year'].astype(str).str.startswith('1938') | 
                  df['Year'].astype(str).str.startswith('1939') | 
                  df['Year'].astype(str).str.startswith('1940') | 
                  df['Year'].astype(str).str.startswith('1941') | 
                  df['Year'].astype(str).str.startswith('1942') | 
                  df['Year'].astype(str).str.startswith('1943'))]

# Extract the 'Quantity withdrawn' column and calculate the mean
withdrawn_avg = filtered_df['Quantity withdrawn'].mean()
print(f"Final Answer: {withdrawn_avg:.1f}")