import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'introduced' and 'retired' to integers or keep as strings for filtering
# We will convert to numeric where possible, but use string comparison for years

# Function to parse year safely
def parse_year(year_str):
    if pd.isna(year_str) or year_str == '-':
        return None
    return int(year_str)

# Apply parsing
df['introduced'] = df['introduced'].apply(parse_year)
df['retired'] = df['retired'].apply(parse_year)

# Define the period: 2004 to 2008
# Fleet in 2004: introduced <= 2004 and retired > 2004 or retired is missing (i.e., retired > 2004 or retired is None)
# Fleet in 2008: introduced >= 2004 and retired > 2008 or retired is missing

# Total seating in 2004
mask_2004 = (df['introduced'] <= 2004) & (
    (df['retired'] > 2004) | (df['retired'].isna()) | (df['retired'] == None)
)
seating_2004 = df.loc[mask_2004, 'seating'].sum()

# Total seating in 2008
mask_2008 = (df['introduced'] >= 2004) & (
    (df['retired'] > 2008) | (df['retired'].isna()) | (df['retired'] == None)
)
seating_2008 = df.loc[mask_2008, 'seating'].sum()

# Change in total seating capacity = seating_2008 - seating_2004
change = seating_2008 - seating_2004

print(f"Final Answer: {change}")