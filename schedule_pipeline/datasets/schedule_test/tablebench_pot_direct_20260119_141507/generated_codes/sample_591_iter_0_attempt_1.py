import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Extract the relevant rows (from 1950-1955 to 1975-1980)
relevant_periods = df[(df['Period'].str.startswith('1950')) | 
                      (df['Period'].str.startswith('1955')) | 
                      (df['Period'].str.startswith('1960')) | 
                      (df['Period'].str.startswith('1965')) | 
                      (df['Period'].str.startswith('1970')) | 
                      (df['Period'].str.startswith('1975'))]

# Clean the 'Live births per year' column by removing spaces and converting to int
df['Live births per year'] = df['Live births per year'].str.replace(' ', '').astype(int)

# Sum the live births for the selected periods
total_live_births = relevant_periods['Live births per year'].sum()

print(f"Final Answer: {total_live_births}")