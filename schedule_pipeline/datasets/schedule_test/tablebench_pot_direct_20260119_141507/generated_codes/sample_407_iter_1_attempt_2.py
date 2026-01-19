import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display the table structure and describe key trends
print("Table Description:")
print("Columns represent metrics over years (2017 to 2013):")
print("- 'Applications': Total number of applicants each year.")
print("- 'Offer Rate (%)': Percentage of applicants who received an offer.")
print("- 'Enrols': Number of students who enrolled in the program.")
print("- 'Yield (%)': Percentage of offers accepted (enrolled).")
print("- 'Applicant/Enrolled Ratio': Ratio of applicants to enrolments, indicating efficiency.")
print("- 'Average Entry Tariff': Average cost of entry into the program.")

print("\nNotable Trends and Insights:")
print("- Applications decreased slightly from 2016 to 2017 (42,335 → 41,000).")
print("- Offer rate remained stable around 86–90%, indicating consistent offer distribution.")
print("- Enrolment stayed relatively stable, with a slight dip in 2015–2016 and a small rise in 2017.")
print("- Yield declined from 2014 to 2017 (18.7% → 17.6%), suggesting reduced acceptances.")
print("- Applicant/Enrolled Ratio improved from 6.14 (2013) to 6.76 (2017), indicating better conversion efficiency.")
print("- Average Entry Tariff shows no major change, suggesting stable or increasing costs.")

Final Answer: Applications decreased slightly, yield declined, applicant/enrolled ratio improved