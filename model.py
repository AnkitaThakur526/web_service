import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler

def train_and_save_model(csv_filename, pickle_filename):
    # Load the dataset
    data = pd.read_csv(csv_filename)
    
    # Encode categorical features
    label_enc = LabelEncoder()
    data['sex'] = label_enc.fit_transform(data['sex'])

    exercise_enc = LabelEncoder()
    data['cp'] = exercise_enc.fit_transform(data['cp'])  # 'cp' example if it’s categorical, adjust if necessary

    # Split the data into features and target
    X = data.drop('target', axis=1)
    y = data['target']

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Naive Bayes model
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)

    # Save the model to a pickle file
    with open(pickle_filename, 'wb') as model_file:
        pickle.dump(nb_model, model_file)
    
    # Save the encoders and scaler to pickle files
    with open('label_encoders.pkl', 'wb') as enc_file:
        pickle.dump((label_enc, exercise_enc), enc_file)

    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    
    print(f"Model saved to '{pickle_filename}'")
    print("Encoders and scaler saved to 'label_encoders.pkl' and 'scaler.pkl'")

# Example usage
if __name__ == "__main__":
    csv_filename = 'synthetic_health_data.csv'
    pickle_filename = 'naive_bayes_model.pkl'
    train_and_save_model(csv_filename, pickle_filename)
