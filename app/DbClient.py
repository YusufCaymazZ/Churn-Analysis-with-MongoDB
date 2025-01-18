#DATA TASKS
from pymongo import MongoClient
from dotenv import load_dotenv  # type: ignore
import os


load_dotenv()

class DbClient:

    # DEFINE VARIABLES
    def __init__(self, lake = os.getenv("LAKE"), warehouse = os.getenv("WAREHOUSE"), 
                predicted_data = os.getenv("PREDICTED_DB"), 
                client = os.getenv("URL"), 
                database = os.getenv("DATABASE"),
                csv_data = os.getenv("CSV_DATA")):
        
        self.client = MongoClient(client)
        self.database = self.client[database]
        self.lake = self.database[lake]
        self.warehouse = self.database[warehouse]
        self.predicted_data = self.database[predicted_data]
        self.csv_data = csv_data