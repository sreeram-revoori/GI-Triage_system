import csv

data = """Gender,Age,Hematochezia,Hematemesis,Melena,Syncope_presyncope,unstable_cad,COPD,CRF,risk_for_stress,cirrhosis,asa_nsaids,PPI,Rectal_exam,Systolic_BP,Diastolic_BP,Heart_rate,Hematocrit_pct,hematocrit_drop (if available),Platelets,BUN_mg_dl,Creatinine_mg_dl,INR,Source_of_bleeding,Need_for_urgent_endoscopy,Disposition,No_of_units_pRBCs_during_transport
"""

# Split the string into lines
lines = data.strip().split('\n')

# Write to CSV
with open("C:\Users\sreer\Downloads\output.csv", 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for line in lines:
        row = line.split(',')  # Change this if your delimiter is different
        writer.writerow(row)

print("CSV file created as output.csv")
