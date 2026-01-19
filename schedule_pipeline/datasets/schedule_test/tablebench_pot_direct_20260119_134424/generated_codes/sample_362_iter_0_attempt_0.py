import pandas as pd

df = pd.read_csv('table.csv')

# Extract melting point from the string (first value before '/')
def extract_melting_point(point_str):
    try:
        return float(point_str.split('/')[0].strip().replace('-', '-'))
    except:
        return None

# Apply function to extract melting points
df['melting_point'] = df['melting / boiling point'].apply(extract_melting_point)

# Count agents with melting point below 0
count_below_zero = df[df['melting_point'] < 0].shape[0]
print(f"Final Answer: {count_below_zero}")