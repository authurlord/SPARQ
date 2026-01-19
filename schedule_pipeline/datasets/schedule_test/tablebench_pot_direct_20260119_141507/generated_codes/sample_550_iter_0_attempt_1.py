import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer and sort by Year
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.sort_values('Year').reset_index(drop=True)

# Extract Political Rights scores
political_rights = df['Political Rights'].astype(int)

# Compare each year with the previous one
for i in range(1, len(political_rights)):
    if political_rights[i] - political_rights[i-1] <= -2:
        answer_year = df.iloc[i]['Year']
        break

print(f"Final Answer: {answer_year}")