import pandas as pd

df = pd.read_csv('table.csv')
# Filter for Grammy Awards, Song of the Year category, and Won result
filtered_df = df[(df['Association'] == 'Grammy Awards') & 
                 (df['Category'] == 'Song of the Year') & 
                 (df['Result'] == 'Won')]

# Extract the year
year = filtered_df['Year'].iloc[0]
print(f"Final Answer: {year}")