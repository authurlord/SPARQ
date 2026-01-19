import pandas as pd

df = pd.read_csv('table.csv')

# Function to extract max power from strings like '200/235'
def extract_max_power(power_str):
    if '/' in power_str:
        return max(map(int, power_str.split('/')))
    return int(power_str)

# Apply the function to convert power output to numeric
df['Power output (hp)'] = df['Power output (hp)'].apply(extract_max_power)

# Sort by power output in descending order
ranked_models = df.sort_values(by='Power output (hp)', ascending=False)['Model']

# Print the ranked models
print(f"Final Answer: {', '.join(ranked_models)}")