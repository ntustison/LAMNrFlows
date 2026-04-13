import antstorch

hcp_template_ids = ["hcpaT1Template",
                    "hcpaT2Template",
                    "hcpaFATemplate",
                    "hcpyaT1Template",
                    "hcpyaT2Template",
                    "hcpyaFATemplate",
                    "hcpinterT1Template",
                    "hcpinterT2Template",  
                    "hcpinterFATemplate"]

for i in range(len(hcp_template_ids)):
   print(f"Downloading {hcp_template_ids[i]}")
   antstorch.get_antstorch_data(hcp_template_ids[i])