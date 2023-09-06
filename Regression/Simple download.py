import requests

# URL of the file to download
# file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/china_gdp.csv"
# path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/china_gdp.csv'
# Send an HTTP GET request to the file URL
response = requests.get(file_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Get the file name from the URL
    file_name = file_url.split("/")[-1]
    
    # Open a file in binary mode and write the content of the response to it
    with open(file_name, "wb") as f:
        f.write(response.content)
        
    print("File downloaded successfully.")
else:
    print(f"Failed to download file. Status code: {response.status_code}")
