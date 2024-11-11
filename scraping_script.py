from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import openpyxl
import time
import json
import os

# Set up the Chrome WebDriver using the Service class (Selenium 4)
chrome_service = Service('/Users/michaeldanzi/Desktop/chromedriver-mac-arm64/chromedriver')  # Update with the actual path to chromedriver
options = webdriver.ChromeOptions()
#options.add_argument("--headless")  # Run in headless mode (no GUI)


proxy_ip_port = '123.47.67.09:8080'  # Example: '123.45.67.89:8080'

# Configure Chrome options
chrome_options = webdriver.ChromeOptions()

# Add proxy settings to Chrome
chrome_options.add_argument(f'--proxy-server={proxy_ip_port}')

prefs = {
    "profile.default_content_setting_values.cookies": 1,  # 1 = Allow, 2 = Block
}
chrome_options.add_experimental_option("prefs", prefs)

# Initialize the WebDriver
driver = webdriver.Chrome(service=chrome_service, options=options)

nutrition_data_list = []

file_path = 'shellfish.json'

# Template for the JSON format we want
def get_nutrition_template():
    return {
        "Product Name": None,
        "Price": None,
        "Servings": None,
        "Calories": None,
        "Fat": {
            "Total Fat": None,
            "Saturated Fat": None,
            "Trans Fat": None
        },
        "Cholesterol": None,
        "Sodium": None,
        "Carbs": {
            "Total Carbs": None,
            "Fiber": None,
            "Sugars": None,
            "Added Sugars": None
        },
        "Protein": None,
        "Micronutrients": {
            "Folate": None,
            "Vitamin C": None,
            "Iron": None,
            "Iodine": None,
            "Vitamin A": None,
            "Zinc": None,
            "Calcium": None,
            "Potassium": None,
            "Vitamin D": None
        },
        "Ingredients": None
    }

def append_to_json_file(file_path, new_data):
    # Check if the file exists
    if os.path.exists(file_path):
        # Open the file and load existing data
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []  # If the file is empty, use an empty list
    else:
        existing_data = []  # If the file doesn't exist, start with an empty list
    
    # Append the new data to the existing data
    existing_data.append(new_data)
    
    # Write the updated data back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, indent=4)

# Function to scrape a single URL
def scrape_url(url):
      driver.get(url)
      time.sleep(3)  # Give the page some time to load
   
      soup = BeautifulSoup(driver.page_source, 'html.parser')

      with open('soup_output.txt', 'w', encoding='utf-8') as file:
         file.write(soup.prettify())
   
      # Initialize the template for storing the nutritional data
      nutrition_data = get_nutrition_template()

      # Extract product name and price (example, adjust to your page structure)
      product_title = soup.find('span', {'id': 'productTitle'})
      if product_title:
         nutrition_data["Product Name"] = product_title.get_text().strip()
      
      product_price = soup.find('span', {'class': 'a-offscreen'})

      if product_price:
         nutrition_data["Price"] = product_price.get_text().strip()
      
      # Now extract the nutrition info
      rows = soup.find_all('tr')

      for row in rows:
         cells = row.find_all('td')

         if len(cells) > 1:
               nutrient_info = cells[1].get_text(separator=" ").strip()
               nutrient_parts = nutrient_info.rsplit(' ', 1)

               if len(nutrient_parts) == 2:
                  nutrient_name = nutrient_parts[0].strip()
                  nutrient_value = nutrient_parts[1].strip()
               else:
                  nutrient_name = nutrient_info
                  nutrient_value = None
               
               # Map the extracted nutrient to the correct field in the template
               if "Calories" in nutrient_name:
                  nutrition_data["Calories"] = nutrient_value
               elif "Total Fat" in nutrient_name:
                  nutrition_data["Fat"]["Total Fat"] = nutrient_value
               elif "Saturated Fat" in nutrient_name:
                  nutrition_data["Fat"]["Saturated Fat"] = nutrient_value
               elif "Trans Fat" in nutrient_name:
                  nutrition_data["Fat"]["Trans Fat"] = nutrient_value
               elif "Cholesterol" in nutrient_name:
                  nutrition_data["Cholesterol"] = nutrient_value
               elif "Sodium" in nutrient_name:
                  nutrition_data["Sodium"] = nutrient_value
               elif "Total Carbohydrate" in nutrient_name:
                  nutrition_data["Carbs"]["Total Carbs"] = nutrient_value
               elif "Dietary Fiber" in nutrient_name:
                  nutrition_data["Carbs"]["Fiber"] = nutrient_value
               elif "Sugars" in nutrient_name and "Added" not in nutrient_name:
                  nutrition_data["Carbs"]["Sugars"] = nutrient_value
               elif "Added Sugars" in nutrient_name:
                  nutrition_data["Carbs"]["Added Sugars"] = nutrient_value
               elif "Protein" in nutrient_name:
                  nutrition_data["Protein"] = nutrient_value
               elif "Folate" in nutrient_name:
                  nutrition_data["Micronutrients"]["Folate"] = nutrient_value
               elif "Vitamin C" in nutrient_name:
                  nutrition_data["Micronutrients"]["Vitamin C"] = nutrient_value
               elif "Iron" in nutrient_name:
                  nutrition_data["Micronutrients"]["Iron"] = nutrient_value
               elif "Iodine" in nutrient_name:
                  nutrition_data["Micronutrients"]["Iodine"] = nutrient_value
               elif "Vitamin A" in nutrient_name:
                  nutrition_data["Micronutrients"]["Vitamin A"] = nutrient_value
               elif "Zinc" in nutrient_name:
                  nutrition_data["Micronutrients"]["Zinc"] = nutrient_value
               elif "Calcium" in nutrient_name:
                  nutrition_data["Micronutrients"]["Calcium"] = nutrient_value
               elif "Potassium" in nutrient_name:
                  nutrition_data["Micronutrients"]["Potassium"] = nutrient_value
               elif "Vitamin D" in nutrient_name:
                  nutrition_data["Micronutrients"]["Vitamin D"] = nutrient_value

      ingredients_section = soup.find('div', {'id': 'ingredients-container'})
      if ingredients_section:
         ingredient_text = ingredients_section.find('span', class_='a-size-small')
         if ingredient_text:
               nutrition_data["Ingredients"] = ingredient_text.get_text().strip()

      if nutrition_data["Ingredients"] is None:
         ingredients_section = soup.find('div', {'id': 'nic-ingredients-content'})

         if ingredients_section:
            ingredient_text = ingredients_section.find('span', class_='a-size-base').get_text().strip()
            nutrition_data["Ingredients"] = ingredient_text
      
      serving_size_section = soup.find('div', {'id': 'alm-nutrition-summary-header'})
      if serving_size_section:
         serving_size_text = serving_size_section.find('span', class_='a-size-mini').get_text().strip()
         nutrition_data["Servings"] = serving_size_text
      
      if nutrition_data["Servings"] is None:
         total_servings_row = soup.find('tr', {'id': 'nic-nutrition-facts-total-serving'})
         serving_size_text = ""

         # Extract the 'servings per container' text
         if total_servings_row:
            total_servings = total_servings_row.find('span', class_='a-size-base')
            if total_servings:
               serving_size_text = total_servings.get_text().strip()

         # Find the serving size row and append the value from the right side
         serving_size_row = soup.find('tr', {'id': 'nic-nutrition-facts-serving-size'})
         if serving_size_row:
            serving_size_value = serving_size_row.find('td', class_='a-text-right').find('span').get_text().strip()
            if serving_size_value:
               # Append the serving size value to the 'servings per container' text
               serving_size_text += " | " + serving_size_value

         # Update the nutrition_data with the final serving size
         nutrition_data["Servings"] = serving_size_text

      
      rows = soup.find_all('tr')

# Loop through each row
      for row in rows:
         left_cell = row.find('td', class_='a-text-left')
         
         if left_cell:
            # Check if the left cell contains 'Calories'
            if 'Calories' in left_cell.get_text():
                  # Extract the right-hand value from the corresponding right cell
                  right_cell = row.find('td', class_='a-text-right')
                  if right_cell:
                     calorie_value = right_cell.find('span').get_text().strip()
                     nutrition_data["Calories"] = calorie_value
                     break

      # Save the nutrition data to a text file in JSON format

      nutrition_data_list.append(nutrition_data)


      return nutrition_data


# Example URL list
urls = [
   "https://www.amazon.com/Whole-Catch-Frozen-Shrimp-Deveined/dp/B084LNPHSX?crid=182JJHGZQLPUC&dib=eyJ2IjoiMSJ9.NnpobIaX22yCpWIb0kPacNbBc9CqW0rgjWOkdRU_Neqi25B8-_ChSHPG9kxDUWKF1pSRMtzbLGeBwphCvbUvkKMHoX4mxrm1Lsva7g2qZGv-nHHqAP69KN3HxiYwAUHI1llXHPccu6llXF2kipaN2_9MovEZ30fYORTj9b4fu5EsrKsXAhkrc7ENJ-MOJURbP5m65w9GD7nf-KMBfH1iJcuJWjfELu9hZefo32uF2OK6qv4j_ITsOQFFFfjQlBB4KLtOVyAQaTmYwkZBGR0ZqOAYYsA5V487VG-IG_kbiKQ.an3pC7Ul-7OK7ly9qiDMCta6acmloi6SdRqpLA6-ja0&dib_tag=se&keywords=shrimp&qid=1731111433&s=grocery&sprefix=shrimp%2Cgrocery%2C112&sr=1-4",
   "https://www.amazon.com/365-Everyday-Value-Caught-Shrimp/dp/B07NR73MGF?crid=182JJHGZQLPUC&dib=eyJ2IjoiMSJ9.NnpobIaX22yCpWIb0kPacNbBc9CqW0rgjWOkdRU_Neqi25B8-_ChSHPG9kxDUWKF1pSRMtzbLGeBwphCvbUvkKMHoX4mxrm1Lsva7g2qZGv-nHHqAP69KN3HxiYwAUHI1llXHPccu6llXF2kipaN2_9MovEZ30fYORTj9b4fu5EsrKsXAhkrc7ENJ-MOJURbP5m65w9GD7nf-KMBfH1iJcuJWjfELu9hZefo32uF2OK6qv4j_ITsOQFFFfjQlBB4KLtOVyAQaTmYwkZBGR0ZqOAYYsA5V487VG-IG_kbiKQ.an3pC7Ul-7OK7ly9qiDMCta6acmloi6SdRqpLA6-ja0&dib_tag=se&keywords=shrimp&qid=1731111433&s=grocery&sprefix=shrimp%2Cgrocery%2C112&sr=1-7",
   "https://www.amazon.com/Whole-Catch-Frozen-Shrimp-Uncooked/dp/B084LNLC55?crid=182JJHGZQLPUC&dib=eyJ2IjoiMSJ9.NnpobIaX22yCpWIb0kPacNbBc9CqW0rgjWOkdRU_Neqi25B8-_ChSHPG9kxDUWKF1pSRMtzbLGeBwphCvbUvkKMHoX4mxrm1Lsva7g2qZGv-nHHqAP69KN3HxiYwAUHI1llXHPccu6llXF2kipaN2_9MovEZ30fYORTj9b4fu5EsrKsXAhkrc7ENJ-MOJURbP5m65w9GD7nf-KMBfH1iJcuJWjfELu9hZefo32uF2OK6qv4j_ITsOQFFFfjQlBB4KLtOVyAQaTmYwkZBGR0ZqOAYYsA5V487VG-IG_kbiKQ.an3pC7Ul-7OK7ly9qiDMCta6acmloi6SdRqpLA6-ja0&dib_tag=se&keywords=shrimp&qid=1731111433&s=grocery&sprefix=shrimp%2Cgrocery%2C112&sr=1-9",
   "https://www.amazon.com/365-WFM-Agedama-Shrimp-Value/dp/B08FN9YWGP?crid=182JJHGZQLPUC&dib=eyJ2IjoiMSJ9.NnpobIaX22yCpWIb0kPacNbBc9CqW0rgjWOkdRU_Neqi25B8-_ChSHPG9kxDUWKF1pSRMtzbLGeBwphCvbUvkKMHoX4mxrm1Lsva7g2qZGv-nHHqAP69KN3HxiYwAUHI1llXHPccu6llXF2kipaN2_9MovEZ30fYORTj9b4fu5EsrKsXAhkrc7ENJ-MOJURbP5m65w9GD7nf-KMBfH1iJcuJWjfELu9hZefo32uF2OK6qv4j_ITsOQFFFfjQlBB4KLtOVyAQaTmYwkZBGR0ZqOAYYsA5V487VG-IG_kbiKQ.an3pC7Ul-7OK7ly9qiDMCta6acmloi6SdRqpLA6-ja0&dib_tag=se&keywords=shrimp&qid=1731111433&s=grocery&sprefix=shrimp%2Cgrocery%2C112&sr=1-19",
   "https://www.amazon.com/Bumble-Bee-dinirao-White-Crabmeat/dp/B000VDWRHU?crid=1U48OIMKQJASW&dib=eyJ2IjoiMSJ9.Msfv9WXWdpsnE6ACUJEwcO-USqm4_1wYU0iGSWWzL_KQvWq6N3b2gkZaLkYy7TW-75_NglotgXQntZZAKDU6zmwe0KWEHOJaAMybCQtp9vHDkXNqFaZwsGn5OrILC-wihquQv1OgwpH2WtG_If0caRSmg5Q02Ot3k4oX6czTQ_1JCLUimP-NqwEY7i8Lj7tUKSNasNN78Vjl1Qw-DgyvVJpC11UqqBi25_SGdy9H7BCUt3xyA6H_7KceXtD5pf-sEBW_PgO5zYLrXLk-Brcwurg7rp-_toefFLXbvdM5ERg.XRQ7q1yGU6-8rELrtjpjhYfBKqSVmfHYjcuebT5VvAs&dib_tag=se&keywords=shell+fish&qid=1731111495&rdc=1&s=grocery&sprefix=shell+fish%2Cgrocery%2C161&sr=1-5",
   "https://www.amazon.com/Bumble-Bee-Clams-Fancy-Whole/dp/B000RUNTH4?crid=1U48OIMKQJASW&dib=eyJ2IjoiMSJ9.Msfv9WXWdpsnE6ACUJEwcO-USqm4_1wYU0iGSWWzL_KQvWq6N3b2gkZaLkYy7TW-75_NglotgXQntZZAKDU6zmwe0KWEHOJaAMybCQtp9vHDkXNqFaZwsGn5OrILC-wihquQv1OgwpH2WtG_If0caRSmg5Q02Ot3k4oX6czTQ_1JCLUimP-NqwEY7i8Lj7tUKSNasNN78Vjl1Qw-DgyvVJpC11UqqBi25_SGdy9H7BCUt3xyA6H_7KceXtD5pf-sEBW_PgO5zYLrXLk-Brcwurg7rp-_toefFLXbvdM5ERg.XRQ7q1yGU6-8rELrtjpjhYfBKqSVmfHYjcuebT5VvAs&dib_tag=se&keywords=shell+fish&qid=1731111495&rdc=1&s=grocery&sprefix=shell+fish%2Cgrocery%2C161&sr=1-2",
   "https://www.amazon.com/Bar-Harbor-Whole-Gourmet-Clams/dp/B087DP99FH?crid=1U48OIMKQJASW&dib=eyJ2IjoiMSJ9.Msfv9WXWdpsnE6ACUJEwcO-USqm4_1wYU0iGSWWzL_KQvWq6N3b2gkZaLkYy7TW-75_NglotgXQntZZAKDU6zmwe0KWEHOJaAMybCQtp9vHDkXNqFaZwsGn5OrILC-wihquQv1OgwpH2WtG_If0caRSmg5Q02Ot3k4oX6czTQ_1JCLUimP-NqwEY7i8Lj7tUKSNasNN78Vjl1Qw-DgyvVJpC11UqqBi25_SGdy9H7BCUt3xyA6H_7KceXtD5pf-sEBW_PgO5zYLrXLk-Brcwurg7rp-_toefFLXbvdM5ERg.XRQ7q1yGU6-8rELrtjpjhYfBKqSVmfHYjcuebT5VvAs&dib_tag=se&keywords=shell+fish&qid=1731111495&s=grocery&sprefix=shell+fish%2Cgrocery%2C161&sr=1-8",
   "https://www.amazon.com/Lukes-Lobster-Secret-Seasoning-Frozen/dp/B07M8HQV2P?crid=1U48OIMKQJASW&dib=eyJ2IjoiMSJ9.Msfv9WXWdpsnE6ACUJEwcO-USqm4_1wYU0iGSWWzL_KQvWq6N3b2gkZaLkYy7TW-75_NglotgXQntZZAKDU6zmwe0KWEHOJaAMybCQtp9vHDkXNqFaZwsGn5OrILC-wihquQv1OgwpH2WtG_If0caRSmg5Q02Ot3k4oX6czTQ_1JCLUimP-NqwEY7i8Lj7tUKSNasNN78Vjl1Qw-DgyvVJpC11UqqBi25_SGdy9H7BCUt3xyA6H_7KceXtD5pf-sEBW_PgO5zYLrXLk-Brcwurg7rp-_toefFLXbvdM5ERg.XRQ7q1yGU6-8rELrtjpjhYfBKqSVmfHYjcuebT5VvAs&dib_tag=se&keywords=shell+fish&qid=1731111495&s=grocery&sprefix=shell+fish%2Cgrocery%2C161&sr=1-23",
   "https://www.amazon.com/Scout-Lobster-Atlantic-Canadian-Ounce/dp/B087KF2VW1?crid=1U48OIMKQJASW&dib=eyJ2IjoiMSJ9.Msfv9WXWdpsnE6ACUJEwcO-USqm4_1wYU0iGSWWzL_KQvWq6N3b2gkZaLkYy7TW-75_NglotgXQntZZAKDU6zmwe0KWEHOJaAMybCQtp9vHDkXNqFaZwsGn5OrILC-wihquQv1OgwpH2WtG_If0caRSmg5Q02Ot3k4oX6czTQ_1JCLUimP-NqwEY7i8Lj7tUKSNasNN78Vjl1Qw-DgyvVJpC11UqqBi25_SGdy9H7BCUt3xyA6H_7KceXtD5pf-sEBW_PgO5zYLrXLk-Brcwurg7rp-_toefFLXbvdM5ERg.XRQ7q1yGU6-8rELrtjpjhYfBKqSVmfHYjcuebT5VvAs&dib_tag=se&keywords=shell+fish&qid=1731111495&s=grocery&sprefix=shell+fish%2Cgrocery%2C161&sr=1-28",
   
   ]

# Scrape the URL and print the nutrition data
for url in urls:
    data = scrape_url(url)
    print(data['Product Name'])

with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(nutrition_data_list, file, indent=4)

# Close the WebDriver
driver.quit()
