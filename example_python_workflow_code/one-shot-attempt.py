# At the start of your notebook:
!pip install requests beautifulsoup4 crewai crewai-tools openai

# Import necessary packages for API key handling
from google.colab import userdata
import os
from openai import OpenAI

# Try multiple methods to set the API key
try:
    # First try Colab secrets
    OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
except:
    # If that fails, ask for manual input
    print("Could not find API key in Colab secrets.")
    OPENAI_API_KEY = input("Please enter your OpenAI API key: ")

# Set the API key
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

print("OpenAI API key has been set successfully!")