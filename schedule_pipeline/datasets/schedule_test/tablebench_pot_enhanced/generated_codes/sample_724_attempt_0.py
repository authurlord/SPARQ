import pandas as pd

df = pd.read_csv('table.csv')

# Function to extract maximum horsepower from strings like '200/235'
def extract_max_hp(hp_str):
    if '/' in hp_str:
        return max(map(int, hp_str.split('/')))
    return int(hp_str)

# Apply the function to convert power output to numeric values
df['Power output (hp)'] = df['Power output (hp)'].apply(extract_max_hp)

# Sort by power output in descending order
ranked_models = df.sort_values(by='Power output (hp)', ascending=False)['Model']

# Print the ranked models
print(f"Final Answer: {', '.join(ranked_models)}")