#DATA TASKS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#MODELING 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.layers import Dropout# type: ignore
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import Callback# type: ignore
from tensorflow.keras.models import load_model# type: ignore

import tensorflow as tf

from churns import Churns
from DbClient import DbClient




class AnalysisTools:
    def __init__(self):
        self.to_collection = DbClient()

    def summerize_model_history(self, model_history):
        
        """
        Model eğitim geçmişini görselleştiren bir fonksiyon.
        `model_history` doğrudan `classifier.fit()` çıktısıdır.
        """
        sns.set(style="whitegrid")  # Seaborn stili ayarlanıyor
        
        # `model_history.history` verilerini işleme
        history = model_history
        epochs = range(1, len(history['loss']) + 1)
        
        # DataFrame oluşturuluyor
        data = {
            'Epoch': list(epochs) + list(epochs),
            'Loss': history['loss'] + history['val_loss'],
            'Type': ['Train'] * len(epochs) + ['Validation'] * len(epochs)
        }
        df = pd.DataFrame(data)
        
        # Seaborn ile görselleştirme
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Epoch', y='Loss', hue='Type', style='Type', markers=True)
        plt.title('Model Loss Over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(title='Type')
        plt.tight_layout()
        plt.savefig("figz/summerized_model_history_seaborn.png")
        plt.show()


    def summerize_loss(self, model_history):
        sns.set(style="whitegrid")  # Seaborn stili
        
        # DataFrame oluşturuluyor
        history = model_history
        epochs = range(1, len(history['loss']) + 1)
        data = {
            'Epoch': list(epochs) + list(epochs),
            'Loss': history['loss'] + history['val_loss'],
            'Type': ['Train'] * len(epochs) + ['Validation'] * len(epochs)
        }
        df = pd.DataFrame(data)
        
        # Seaborn ile çizim
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Epoch', y='Loss', hue='Type', style='Type', markers=True)
        plt.title('Model Loss Over Epochs')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(title='Type')
        plt.tight_layout()
        plt.savefig("figz/summerized_loss_seaborn.png")
        plt.show()

    def weight_visualize(self, weights):
        sns.set(style="white")  # Daha sade bir görünüm için beyaz stil
    
        for i in range(0, len(weights), 2):  # Weights ve biases çiftleri
            weight_matrix = weights[i]
            bias_vector = weights[i + 1]
            
            # Heatmap for weights
            plt.figure(figsize=(10, 6))
            sns.heatmap(weight_matrix, cmap=sns.cubehelix_palette(as_cmap=True))

            plt.title(f"Heatmap of Weights in Layer {i//2 + 1}")
            plt.tight_layout()
            plt.savefig(f"figz/Heatmap-of-Weights-in-Layer-{i//2 + 1}.png")
            plt.show()
            
            # Histogram for biases
            plt.figure(figsize=(8, 5))
            sns.set_theme(style="ticks")
            sns.color_palette("flare", as_cmap=True)
            sns.histplot(bias_vector, kde=True,fill=False, bins=20)
            plt.title(f"Biases of Layer {i//2 + 1}")
            plt.xlabel("Index")
            plt.ylabel("Bias Value")
            plt.tight_layout()
            plt.savefig(f"figz/Biases-of-Layer-{i//2 + 1}.png")
            plt.show()

    def analys_data(self):
        client = self.to_collection.client
        db = self.to_collection.database
        collection_lake = self.to_collection.lake

        data = list(collection_lake.find())
        for document in data:
            document.pop("_id", None)
        df = pd.DataFrame(data)
        X = df.iloc[:, 3:13]
        y = df.iloc[:, 13]
        geography = pd.get_dummies(X['Geography'], drop_first=True)
        gender = pd.get_dummies(X['Gender'], drop_first=True)
        X = X.drop(['Geography', 'Gender'], axis=1)
        X = pd.concat([X, geography, gender], axis=1)

        # X = ANALYZED DATA
        analyzeddata = pd.concat([X, y], ignore_index=True)
        churn = Churns()
        churn.send_to_warehouse(analyzeddata)
        print("\nData sent to warehouse YAY!")

        # SPLIT
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # FEATURE SCALING
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # ANN
        classifier = Sequential()  # -> INIT
        classifier.add(Dense(units=11, activation='relu'))  # -> INPUT
        classifier.add(Dense(units=7, activation='relu'))  # -> HIDDEN
        classifier.add(Dropout(0.2))
        classifier.add(Dense(units=6, activation='relu'))  # -> HIDDEN
        classifier.add(Dropout(0.3))
        classifier.add(Dense(1, activation='sigmoid'))  # -> OUTPUT

        best_epoch_saver = BestEpochSaver()
        opt= tf.keras.optimizers.Adam(learning_rate=0.005)
        classifier.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
        

        
        model_history = classifier.fit(
            X_train,
            y_train,
            validation_split=0.33,
            batch_size=10,
            epochs=64,
            callbacks=[best_epoch_saver]
        )        
        

        # Save the model history
        with open('models/model_history.pkl', 'wb') as f:
            pickle.dump(model_history.history, f)
        
        # Load the best model from file
        best_model = load_model(f"models/best_epoch_{best_epoch_saver.best_epoch + 1}.h5")
        # best_model = load_model(f"models/best_epoch_33.h5")


        # Visualizing model history
        with open('models/model_history.pkl', 'rb') as f:
            history = pickle.load(f)

        self.summerize_model_history(history)
        self.summerize_loss(history)

        # Predicting the Test set results
        y_pred = best_model.predict(X_test)
        y_pred = (y_pred >= 0.5).astype(int)  # Convert probabilities to binary (0 or 1)
        print(y_pred, y_test)
        if isinstance(y_test, np.ndarray):
            y_test = pd.Series(y_test)

        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)

        # Reset indices for alignment
        X_test_reset = X_test.reset_index(drop=True)
        y_test_reset = y_test.reset_index(drop=True)


        # Create results DataFrame and add Actual and Predicted columns
        results = X_test_reset.copy()
        results['Actual'] = y_test_reset
        results['Predicted'] = y_pred
        print(results.head())
        resultsToSend = results[['Actual','Predicted']]
        churn.send_to_predicted_data(pd.DataFrame(resultsToSend))
        print("\nData sent to Predicted-Data database. YAAAAAAAAAAY!")

        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        score = accuracy_score(y_pred, y_test)
        print(score)
        weights = best_model.get_weights()
        self.weight_visualize(weights)


        accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
        print(f'\nAccuracy: {accuracy}')

        precision = cm[1][1] / (cm[0][1] + cm[1][1])
        print(f'\nPrecision: {precision}')

        recall = cm[1][1] / (cm[1][0] + cm[1][1])
        print(f'\nRecall: {recall}')

        f1_score = 2 * (precision * recall) / (precision + recall)
        print(f'\nF1 Score: {f1_score}')


class BestEpochSaver(Callback):
    def __init__(self):
        super().__init__()
        self.best_accuracy = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # Get validation accuracy
        val_accuracy = logs.get("val_accuracy")
        if val_accuracy and val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_epoch = epoch
            self.model.save(f"models/best_epoch_{epoch + 1}.h5")
            print(f"\nSaved model from epoch {epoch + 1} with val_accuracy: {val_accuracy:.4f}")
