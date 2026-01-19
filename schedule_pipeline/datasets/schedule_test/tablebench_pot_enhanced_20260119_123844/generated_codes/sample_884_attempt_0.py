import pandas as pd

df = pd.read_csv('table.csv')

# Convert necessary columns to numeric
df['domestic s ton'] = pd.to_numeric(df['domestic s ton'])
df['foreign total s ton'] = pd.to_numeric(df['foreign total s ton'])

# Initialize list to store results
results = []

# Loop through years starting from 2001 (index 1) to compare with previous year
for i in range(1, len(df)):
    prev_domestic = df.iloc[i-1]['domestic s ton']
    curr_domestic = df.iloc[i]['domestic s ton']
    prev_foreign = df.iloc[i-1]['foreign total s ton']
    curr_foreign = df.iloc[i]['foreign total s ton']
    
    # Calculate percentage increase
    dom_increase = ((curr_domestic - prev_domestic) / prev_domestic) * 100
    foreign_increase = ((curr_foreign - prev_foreign) / prev_foreign) * 100
    
    # Check if domestic increase > foreign total increase
    if dom_increase > foreign_increase:
        results.append(df.iloc[i]['year'])

# Output the year(s) where domestic increase > foreign total increase
if results:
    print(f"Final Answer: {results[0]}")
else:
    print("Final Answer: None")