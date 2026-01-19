import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Identify the row where the category is "Literates"
# The first column is '-' and the second is 'Total', so we look for the row where the value is 'Literates'
# Since the first column is '-', we find the row where the second column (index 1) is 'Literates'
literates_row = df[df['Total'] == 'Literates']

# If found, extract Male and Female values
if not literates_row.empty:
    male_literates = int(literates_row['Male'].values[0])
    female_literates = int(literates_row['Female'].values[0])
    
    if male_literates > female_literates:
        print("Final Answer: Male")
    else:
        print("Final Answer: Female")
else:
    # Fallback: if no match, try to locate by index or name
    # In this case, the data structure suggests that "Literates" is in the second row
    # So we manually index to the correct row
    row_index = df[df.iloc[:, 0] == 'Literates'].index[0] if df[df.iloc[:, 0] == 'Literates'].empty == False else None
    if row_index is not None:
        male_literates = int(df.loc[row_index, 'Male'])
        female_literates = int(df.loc[row_index, 'Female'])
        if male_literates > female_literates:
            print("Final Answer: Male")
        else:
            print("Final Answer: Female")
    else:
        print("Final Answer: Unknown")