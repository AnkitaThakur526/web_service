import numpy as np
import pandas as pd

# Parameters for synthetic data
np.random.seed(42)
n_samples = 1000

# Generate synthetic data
age = np.random.randint(30, 80, n_samples)
sex = np.random.choice(['Male', 'Female'], n_samples)
cp = np.random.choice([0, 1, 2, 3], n_samples)  # Example: 0 = typical angina, 1 = atypical angina, etc.
trestbps = np.random.normal(130, 20, n_samples)
chol = np.random.normal(200, 50, n_samples)
fbs = np.random.choice([0, 1], n_samples)  # 0 = normal, 1 = high
restecg = np.random.choice([0, 1, 2], n_samples)  # 0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy
thalach = np.random.normal(150, 20, n_samples)
exang = np.random.choice([0, 1], n_samples)  # 0 = no, 1 = yes
oldpeak = np.random.normal(1.0, 1.0, n_samples)
slope = np.random.choice([0, 1, 2], n_samples)  # 0 = upsloping, 1 = flat, 2 = downsloping
ca = np.random.choice([0, 1, 2, 3], n_samples)  # Number of major vessels colored by fluoroscopy
thal = np.random.choice([0, 1, 2, 3], n_samples)  # 1 = normal, 2 = fixed defect, 3 = reversable defect

# Generate condition likelihoods
condition_likelihood = (0.4 * (age > 60) +
                        0.3 * (trestbps > 140) +
                        0.3 * (chol > 240) +
                        0.2 * (fbs == 1)).astype(float)

condition_likelihood += np.random.normal(0, 0.1, n_samples)  # Added noise

# Define multi-class conditions
conditions = np.select(
    [
        condition_likelihood < 0.2,
        (condition_likelihood >= 0.2) & (condition_likelihood < 0.5),
        (condition_likelihood >= 0.5) & (condition_likelihood < 0.8),
        condition_likelihood >= 0.8
    ],
    [0, 1, 2, 3]
)

# Create DataFrame
data = pd.DataFrame({
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal,
    'target': conditions
})

# Save DataFrame to CSV
csv_filename = 'synthetic_health_data.csv'
data.to_csv(csv_filename, index=False)

print(f"CSV file '{csv_filename}' has been created.")
