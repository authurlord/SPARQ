import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Extract the names from 'senior pga championship' and 'senior players championship'
def extract_name(entry):
    if pd.isna(entry) or entry == 'not a champions tour event':
        return None
    if isinstance(entry, str):
        # Split by parentheses and take the first part
        name_part = entry.split(' (')[0]
        return name_part.strip()
    return None

# Apply the function to both columns
df['senior_pga_name'] = df['senior pga championship'].apply(extract_name)
df['senior_players_name'] = df['senior players championship'].apply(extract_name)

# Find years where both names match
matching_years = df[df['senior_pga_name'] == df['senior_players_name']]['year']

# If there are matching years, return the first one (or all if multiple)
if not matching_years.empty:
    final_year = matching_years.iloc[0]
else:
    final_year = None

print(f"Final Answer: {final_year}")