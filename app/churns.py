from DbClient import DbClient
import pandas as pd
class Churns: # DATASET(.csv) MUST HAVE BANKING CHURN CUSTOMER DATASET 
    def __init__(self):
        self.to_collections = DbClient()
        self.csv_df = pd.read_csv(self.to_collections.csv_data)

    #THIS FUNCTION WILL USED FOR SEND TO LAKE
    def send_to_lake(self):
        data = self.csv_df

        records = data.to_dict(orient='records')
        if records:
            self.sender_to_lake(records)
        else:
            print("CSV dosyası boş veya hatalı!")
        

    # RAW DATA SHOULD BE SEND TO LAKE COLLECTION
    def sender_to_lake(self,records):
        client = self.to_collections.client
        db = self.to_collections.database
        collection = self.to_collections.lake
        
        collection.insert_many(records)
        print(f"{len(records)} kayıt başarıyla {collection.name} koleksiyonuna eklendi.")
    
    
    
    #THIS FUNCTION WILL CLEAR ALL THE LAKE COLLECTION 
    def clear_lake(self):
        client = self.to_collections.client
        db = self.to_collections.database
        collection = self.to_collections.lake
        collection.drop()

    

    # ANALYSED DATA SHOULD BE SEND TO WAREHOUSE COLLECTION   
    def send_to_warehouse(self, analyzed_data):
            # Ensure analyzed_data is a DataFrame
        if isinstance(analyzed_data, pd.DataFrame):
            # Reset the index to avoid numeric index causing issues
            analyzed_data = analyzed_data.reset_index(drop=True)
            
            # Ensure all column names are strings (important for MongoDB document keys)
            analyzed_data.columns = [str(col) for col in analyzed_data.columns]

            # Convert the DataFrame to a dictionary
            records = analyzed_data.to_dict(orient='records')
            
            if records:
                # Perform insertion into MongoDB
                client = self.to_collections.client
                db = self.to_collections.database
                collection = self.to_collections.warehouse
                
                collection.insert_many(records)
                print(f"{len(records)} kayıt başarıyla {collection.name} koleksiyonuna eklendi.")
            else:
                print("DataFrame boş veya hatalı!")
        else:
            print("Geçersiz veri formatı. Pandas DataFrame bekleniyor.")

    
    def clear_warehouse(self):
        client = self.to_collections.client
        db = self.to_collections.database
        collection = self.to_collections.warehouse
        collection.drop()

    def send_to_predicted_data(self,predicted):

        results = predicted
        records = results.to_dict(orient='records')

        # Insert the records into the 'predicted-data' collection in MongoDB
        if records:
            client = self.to_collections.client
            db = self.to_collections.database
            predicted_collection = self.to_collections.predicted_data
            
            predicted_collection.insert_many(records)
            print(f"{len(records)} predictions have been successfully added to the predicted-data collection.")
        else:
            print("No predictions to insert!")

    def clear_predicted_data(self):

        client = self.to_collections.client
        db = self.to_collections.database
        collection = self.to_collections.predicted_data
        collection.drop()




