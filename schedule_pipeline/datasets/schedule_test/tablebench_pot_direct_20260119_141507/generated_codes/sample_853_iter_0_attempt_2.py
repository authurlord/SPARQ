import pandas as pd

df = pd.read_csv('table.csv')

# Extract the numeric US Chart position by removing text in parentheses
def extract_position(pos):
    try:
        # Remove any text in parentheses and convert to int
        return int(''.join(c for c in pos if c.isdigit()))
    except:
        return 0

df['US Chart position'] = df['US Chart position'].apply(extract_position)

# Find the year with the highest and lowest chart position
max_pos_row = df.loc[df['US Chart position'].idxmax()]
min_pos_row = df.loc[df['US Chart position'].idxmin()]

highest_year = max_pos_row['Year']
lowest_year = min_pos_row['Year']

print(f"Final Answer: {highest_year}, {lowest_year}")