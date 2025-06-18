import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

prefix = '/opt/ml/'

input_path = os.path.join(prefix, 'input', 'data')
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input', 'config', 'hyperparameters.json')

channel_name = 'training'
training_path = os.path.join(input_path, channel_name)

def read_training_data(training_path):
    input_files = [os.path.join(training_path, file) for file in os.listdir(training_path)]
    if not input_files:
        raise ValueError(f"No training files found in: {training_path}")
    data_frames = [pd.read_csv(file) for file in input_files]
    return pd.concat(data_frames)

def prepare_training_data(df):
    X = df.drop(columns=['species'])
    y = df['species']
    return X, y

def run_training(args):
    df = read_training_data(training_path)
    X, y = prepare_training_data(df)

    model = LogisticRegression()
    model.fit(X, y)
    print(f"Training completed. Accuracy: {model.score(X, y):.4f}")

    # ✅ Save model where SageMaker expects it
    model_file = os.path.join(model_path, "model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    args, _ = parser.parse_known_args()
    run_training(args)
