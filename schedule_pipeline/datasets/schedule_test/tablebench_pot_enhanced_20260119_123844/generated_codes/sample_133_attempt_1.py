import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('table.csv')

# Extract the relevant data
interferometer_mode = 'H'
min_baseline = 34
max_baseline = 330

# Create data for plotting
baseline_data = {
    'Minimum Baseline (m)': [min_baseline],
    'Maximum Baseline (m)': [max_baseline]
}

# Convert to DataFrame
baseline_df = pd.DataFrame(baseline_data, index=[interferometer_mode])

# Plot grouped bar chart
baseline_df.plot(kind='bar', figsize=(8, 6), color=['skyblue', 'salmon'])
plt.title('Maximum and Minimum Baselines for Interferometer and Observing Mode H')
plt.xlabel('Interferometer and Observing Mode')
plt.ylabel('Baseline (m)')
plt.xticks(rotation=0)
plt.legend(title='Baseline Type')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

# Final Answer: H
print("Final Answer: H")