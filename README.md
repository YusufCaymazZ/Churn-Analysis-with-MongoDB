# Churn Analysis with MongoDB

## Project Overview

This project analyzes, models, and makes predictions based on a churn dataset with 10,000 data lines. The entire process follows these steps:

1. **Data Import and Preprocessing**: Raw data is imported and stored in MongoDB.
2. **Model Training and Evaluation**: A model is trained and evaluated.
3. **Prediction**: The trained model is used to make churn predictions.
4. **Data Visualization**: Analysis results and model performance are visualized.

After the process, the best epoch model is saved in the `models/` directory along with the training history. The weights and history plots are saved in the `app/figz/` folder. MongoDB collections are used to manage data at different stages.

## Setup and Installation

### Requirements
- Python 3.8+
- MongoDB
- Libraries:
    - pandas
    - numpy
    - matplotlib
    - tensorflow
    - pymongo
    - streamlit
    - pillow
    - scikit-learn
    - python-dotenv # for .env users.


To install the necessary libraries, run:

```bash
pip install -r requirements.txt
```
### MongoDB Setup

Before running the analysis, ensure that MongoDB is set up and running on your system. Follow these steps to configure the required collections:

1. **Start MongoDB**  
   Make sure MongoDB is installed and running. If you haven't installed MongoDB yet, you can follow the installation instructions for your operating system from the [official MongoDB website](https://www.mongodb.com/docs/manual/installation/).

   On most systems, you can start MongoDB by running the following command in your terminal:
   ```bash
   mongod
   ```

2. **Create MongoDB Database**  
   By default, MongoDB will use a database called `test`. You can create a new database for this project by opening the MongoDB shell and running:
   ```bash
   use churn_analysis_db
   ```

3. **Create Collections**  
   This project requires three MongoDB collections. You can create them manually in the MongoDB shell or they will be created automatically when the program runs. However, to ensure that everything is set up correctly, you can create them manually as follows:

   - **`churn-not-analyzed`**: This collection will store the raw data (before any preprocessing).
   - **`churn-analyzed`**: This collection will store the cleaned and preprocessed data.
   - **`churn-predicted`**: This collection will store the predictions made by the model.

   To create these collections manually, run the following commands in the MongoDB shell:

   ```bash
   db.createCollection("churn-not-analyzed")
   db.createCollection("churn-analyzed")
   db.createCollection("churn-predicted")
   ```

4. **Verify Collections**  
   After creating the collections, you can verify their creation by running:
   ```bash
   show collections
   ```

   This should display:
   ```
   churn-not-analyzed
   churn-analyzed
   churn-predicted
   ```

5. **Connect MongoDB to Your Python Script**  
   In your Python script, ensure that the `pymongo` library is used to connect to your local MongoDB instance. If you don't have `pymongo` installed, you can install it by running:
   ```bash
   pip install pymongo
   ```

   In your `main.py` file, use the following code to connect to MongoDB:

   ```python
   from pymongo import MongoClient

   # Connect to MongoDB (default localhost connection)
   client = MongoClient("mongodb://localhost:27017/")
   db = client["churn_analysis_db"]
   ```

   This will connect to your MongoDB instance and use the `churn_analysis_db` database.

Once MongoDB is set up and the collections are created, you can proceed to run the analysis with the data being stored and processed in the corresponding collections.

### Running the Analysis

To run the analysis and ensure everything works correctly, follow these steps:

1. **Prepare Your Data**  
   Ensure that the `Churns.csv` file is placed in the project root directory. This file should contain the churn data with 10,000 data lines.

2. **Check Your MongoDB Connection**  
   Ensure that MongoDB is running and the necessary collections (`churn-not-analyzed`, `churn-analyzed`, `churn-predicted`) are already created. If not, refer to the [MongoDB Setup](#mongodb-setup) section to set up MongoDB.

3. **Run the Python Script**  
   In the project directory, open a terminal and execute the following command to start the analysis:

   ```bash
   python main.py
   ```

   This will run the `main.py` script, which will perform the following actions:

   - **Step 1**: Load the data from `Churns.csv`.
     - The data is imported and transferred to the `churn-not-analyzed` collection in MongoDB.

   - **Step 2**: Preprocess the data.
     - Data cleaning, feature engineering, and any necessary transformations are done.

   - **Step 3**: Insert processed data into the `churn-analyzed` collection.
     - After preprocessing, the cleaned data is stored in the `churn-analyzed` MongoDB collection.

   - **Step 4**: Train the model.
     - A machine learning model (such as a neural network or decision tree) is trained using the processed data. During training, the model will evaluate its performance across multiple epochs.

   - **Step 5**: Save the best epoch model and training history.
     - After training, the best performing model and its history (including loss, accuracy, etc.) will be saved in the `models/` directory.

   - **Step 6**: Make Predictions.
     - Once the model is trained, predictions are made using the trained model.

   - **Step 7**: Store predictions in MongoDB.
     - The churn predictions are stored in the `churn-predicted` collection.

4. **Monitor the Output**  
   During the execution, you can monitor the output in the terminal. The script will print logs such as model training progress, data insertion confirmations, and any potential issues.

5. **Check MongoDB Collections**  
   After the script has finished running, you can check the following MongoDB collections for the respective data:

   - **`churn-not-analyzed`**: Raw data before preprocessing.
   - **`churn-analyzed`**: Cleaned and preprocessed data.
   - **`churn-predicted`**: The churn predictions made by the trained model.

6. **Visualizations**  
   The training history and model weights are saved in the `app/figz/` folder as images. You can view these files to analyze the model's learning process.

   Example plot for training history:
   ![Training History](app/figz/history_plot.png)

7. **Review the Saved Model**  
   The best model is saved in the `models/` directory. This model can be reloaded later to make further predictions or fine-tune it.

Once the analysis is complete, you can review the results, the visualizations, and the saved model to assess its performance and make further improvements as needed.

  ## Visualizations

After the model training process is completed, the training history and model weights are visualized to provide insights into the model's learning performance. These visualizations are stored in the `app/figz/` folder.

### Visualizing Training History

Training history typically includes metrics like accuracy, loss, etc., across each epoch. The visualizations help to understand how the model improved over time and can highlight any issues such as overfitting or underfitting.

To visualize the training history:

1. **Training History Plot**: This plot shows how the model's loss and accuracy change as training progresses. This allows you to see how well the model is fitting to the training data.

   Example of a training history plot:
   
   ![Training History](app/figz/history_plot.png)

2. **Loss vs Epoch Curve**: A curve showing the loss (or error) value at each epoch during training. This can be helpful to see whether the model converges properly or if more epochs are needed.

   Example of a loss vs epoch curve:
   
   ![Epoch vs Loss](app/figz/loss_vs_epoch.png)

### Visualizing Model Weights

Another important aspect is to visualize the model's weights. This can be done by using techniques such as histograms or heatmaps. These visualizations show how the model’s weights change during training and can offer insights into the model's inner workings.

- **Weight Distribution**: A histogram or a bar plot showing the distribution of weights for different layers in the neural network model.

   Example of weight distribution plot:
   
   ![Weight Distribution](app/figz/weight_distribution.png)

- **Heatmap of Weights**: A heatmap of the model's weights provides a visual representation of the weight matrix for each layer.

   Example of heatmap of weights:
   
   ![Weight Heatmap](app/figz/weight_heatmap.png)

### Saving and Viewing the Visualizations

All the plots generated during training (e.g., accuracy, loss, weight distributions) are saved in the `app/figz/` folder in image formats like `.png` or `.jpg`. These can be reviewed to evaluate the model's performance.

You can view the visualizations by opening the respective image files from the `app/figz/` directory.

### Example Commands to Generate Visualizations (Optional)

If you want to generate these plots manually or modify the plotting settings, you can use a Python script similar to the following (example for plotting training history):

```python
import matplotlib.pyplot as plt

# Assume history is a dictionary containing training history, e.g., from a Keras model
history = {'accuracy': [0.1, 0.3, 0.5], 'loss': [0.9, 0.6, 0.4]}  # Example data

plt.plot(history['accuracy'], label='Accuracy')
plt.plot(history['loss'], label='Loss')
plt.title('Training History')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.savefig('app/figz/history_plot.png')
plt.show()
```
This will generate and save the `history_plot.png` file in the `app/figz/` folder.

### Review the Model Performance

These visualizations offer a deeper understanding of the model’s performance throughout training. By carefully analyzing them, you can determine if the model requires further tuning, additional epochs, or other adjustments to improve its accuracy or prevent issues such as overfitting.

If you observe that the model is overfitting (where the training accuracy improves significantly while the validation accuracy remains stagnant or drops), you may need to take actions like:

- Implementing regularization techniques (e.g., **dropout** layers in neural networks).
- Using **data augmentation** to artificially increase the size and variety of your training set.
- Trying **early stopping** to prevent the model from training beyond a certain point.

### Conclusion

Visualizations are an invaluable tool in model evaluation, as they provide clear insights into the model’s learning process and training dynamics. They help identify potential areas for improvement, such as signs of overfitting or underfitting.

You are encouraged to modify the existing plotting scripts or add new types of visualizations (e.g., **confusion matrices**, **ROC curves**) to further analyze the model's performance and make informed decisions about its optimization.

## Review the Saved Model

Once the training process is complete, the best performing model is saved in the `models/` directory. This allows you to store the trained model for future use, such as making new predictions or continuing the training process if needed.

### Checking the Saved Model

The saved model will typically be stored in a format compatible with the framework used (for example, a `.h5` file for Keras/TensorFlow models). You can load the model back at any time to evaluate it further or make predictions on new data.

#### Example of Saving the Model

In the training script, the model is saved using the following code:

```python
from tensorflow.keras.models import save_model

# Assuming 'model' is the trained model object
model.save('models/best_model.h5')
```

This will save the best model (based on performance) in the `models/` directory as `best_model.h5`.

### Loading the Saved Model

To reload the saved model and use it for making predictions or further training, you can use the following code:

```python
from tensorflow.keras.models import load_model

# Load the model
model = load_model('models/best_model.h5')

# Use the model to make predictions
predictions = model.predict(new_data)
```

### Model Evaluation

After loading the saved model, you can re-evaluate it on a test set, compare its performance to earlier evaluations, or even fine-tune it. This allows you to track the model's performance over time and make adjustments as necessary.

### Model Deployment

The saved model can also be deployed to a production environment, where it can handle real-time data and provide predictions on new input. Depending on your application, you can set up an API endpoint for the model, integrate it with a web service, or use it in other real-world applications.

### Conclusion

Saving and loading the trained model is a crucial part of the machine learning lifecycle. It enables you to persist your work, share your model with others, and apply it in different contexts without needing to retrain it every time.

## Future Improvements

While the current model provides useful insights, there are a few areas that can be further improved:
- Experiment with different algorithms and model architectures to see if performance can be enhanced.
- Explore additional data sources or features that could improve the accuracy of churn predictions.
- Implement hyperparameter tuning to optimize model performance.

## Contributing

Contributions to this project are welcome! If you have any suggestions, improvements, or bug fixes, feel free to fork the repository and submit a pull request.

### How to contribute:
1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Make the necessary changes and commit them.
4. Push your changes to your fork and open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



