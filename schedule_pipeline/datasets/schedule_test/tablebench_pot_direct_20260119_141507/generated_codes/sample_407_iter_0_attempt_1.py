import pandas as pd

# Load the table
df = pd.read_csv('table.csv')

# Display the structure and content of the table
print("Column Descriptions and Notable Trends:")
print("-" * 50)
print("Columns and their meanings:")
print(" - 'Applications': Total number of applicants each year.")
print(" - 'Offer Rate (%)': Percentage of applicants who received an offer.")
print(" - 'Enrols': Number of students who actually enrolled each year.")
print(" - 'Yield (%)': Percentage of offers that resulted in enrolments.")
print(" - 'Applicant/Enrolled Ratio': Ratio of applicants to those who enrolled, indicating competitiveness.")
print(" - 'Average Entry Tariff': Average fee paid by enrolled students (n/a in 2017).")

print("\nNotable Trends and Insights:")
print("- Applications show a slight decline from 2016 to 2017 (42,335 → 41,000), but remain relatively stable.")
print("- Offer rate fluctuates slightly, with a drop from 2015 (90.8%) to 2016 (86.4%), then recovering to 89.7% in 2017.")
print("- Enrolments are stable, with a small increase from 2014 to 2015, followed by a slight dip in 2016.")
print("- Yield (%) increases from 2013 to 2014 (18.7% → 17.6%), then drops slightly, suggesting a possible shift in student acceptance.")
print("- Applicant/Enrolled Ratio improves slightly over time (6.14 in 2013 to 6.76 in 2017), indicating increasing competition.")
print("- Average Entry Tariff decreases from 2015 to 2016 (471 → 176), suggesting a significant reduction in fees, though it is missing in 2017.")

Final Answer: Applications declined slightly, offer rate fluctuated, enrolments were stable, yield decreased, applicant ratio increased, tariff dropped significantly