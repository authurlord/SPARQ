import pandas as pd
import re

df = pd.read_csv('table.csv')

# Clean the 'Power output (hp)' column by extracting numeric values
def extract_hp(value):
    if pd.isna(value):
        return 0
    # Handle cases like "200/235" or "300"
    match = re.search(r'(\d+)(?:\/(\d+))?$', str(value))
    if match:
        num1 = int(match.group(1))
        num2 = int(match.group(2)) if match.group(2) else num1
        return max(num1, num2)
    return 0

# Apply the cleaning function
df['Power output (hp)'] = df['Power output (hp)'].apply(extract_hp)

# Sort by power output in descending order and get the rank
ranked_models = df[['Model', 'Power output (hp)']].sort_values(by='Power output (hp)', ascending=False)

# Extract the ranked list of models
result = ranked_models['Model'].tolist()

print(f"Final Answer: {', '.join(result)}")