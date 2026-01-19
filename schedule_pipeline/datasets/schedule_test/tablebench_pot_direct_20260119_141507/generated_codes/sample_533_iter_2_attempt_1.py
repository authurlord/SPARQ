import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# We are interested in the row where SPECIFICATION is "I." and the second column is "TOTAL"
# The row with "TOTAL" population by age group
# Find the row where the first two values are 'I.' and 'TOTAL'
# The full row: ['I.', 'TOTAL', 'TOTAL', 'TOTAL', 'TOTAL', 'Measure unit', 'TOTAL', '0-9', '10-19', ...]

# Filter the row with 'I.' and 'TOTAL' in the first two columns
# Since the header row has 'SPECIFICATION' and then the rest, we can locate the row with 'I.' and 'TOTAL'

# We know the row of interest is the one after the header
# The data starts with a row: ['I.', 'TOTAL', 'TOTAL', 'TOTAL', 'TOTAL', 'person', '287', '41', '45', '47', '27', '38', '31', '20', '24', '14']

# Let's find the row where the first element is 'I.' and the second is 'TOTAL'
# The index of such a row
rows = df.values
for i, row in enumerate(rows):
    if row[0] == 'I.' and row[1] == 'TOTAL':
        # Extract the values for the age groups
        age_0_to_9 = row[8]  # index 8 is '0-9'
        age_80_plus = row[15]  # index 15 is '80 +'
        
        # Age groups 11–59: sum of 10–19, 20–29, 30–39, 40–49, 50–59
        # These are at indices 9 to 13 (since 0-9 is at 8, 10-19 at 9, etc.)
        age_11_to_59 = sum(row[9:14])  # indices 9 to 13 inclusive
        
        # Calculate dependency ratio
        numerator = int(age_0_to_9) + int(age_80_plus)
        denominator = age_11_to_59
        
        if denominator == 0:
            dependency_ratio = float('inf')
        else:
            dependency_ratio = numerator / denominator
            
        print(f"Final Answer: {dependency_ratio:.2f}")
        break