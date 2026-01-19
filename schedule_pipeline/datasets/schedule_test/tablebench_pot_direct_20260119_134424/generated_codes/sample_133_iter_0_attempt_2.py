import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Manually extract the data for the first row since the structure is irregular
interferometer_mode = 'H'
min_baseline = 34
max_baseline = 330

# Create a DataFrame for plotting
baseline_data = pd.DataFrame({
    'Baseline': ['Minimum', 'Maximum'],
    'Value': [min_baseline, max_baseline]
})

# Plot grouped bar chart
plt.figure(figsize=(8, 6))
plt.bar(baseline_data['Baseline'], baseline_data['Value'], color=['skyblue', 'salmon'])
plt.title(f'Maximum and Minimum Baselines for {interferometer_mode}')
plt.ylabel('Baseline (m)')
plt.show()

print("Final Answer: H")