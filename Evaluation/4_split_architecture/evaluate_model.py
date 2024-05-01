import numpy as np
import keras
from keras.models import model_from_json
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Make sure you have assigned the model folder name to val_acc")
print("")
# Set your model folder here
val_acc = 0.9482

# Load the test data for four inputs
X0, Y0 = np.load("NPY_files/prepared_test0_dataset.npy", allow_pickle=True)
X1, Y1 = np.load("NPY_files/prepared_test1_dataset.npy", allow_pickle=True)

# Assuming X0 and X1 are lists of arrays, where each list item corresponds to one part of the split input
input_0 = [X0[i] for i in range(4)]  # Four parts for the first dataset
input_1 = [X1[i] for i in range(4)]  # Four parts for the second dataset

# Print shapes to debug
for idx, inp in enumerate(input_0):
    print(f"Shape of input_0[{idx}]: {inp.shape}")

# Load the model
json_file = open(f'../../Training/4_split_architecture/{val_acc}/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(f'../../Training/4_split_architecture/{val_acc}/model.h5')
print("Loaded model from disk")

# Load batch size from the data.txt
with open(f'../../Training/4_split_architecture/{val_acc}/data.txt') as json_file:
    data = json.load(json_file)
batch_size = data['details']['batch_size']

# Compile the model
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results_0 = loaded_model.evaluate(input_0, Y0, batch_size=batch_size)
results_1 = loaded_model.evaluate(input_1, Y1, batch_size=batch_size)

print('test loss_0, test acc_0:', results_0)
print('test loss_1, test acc_1:', results_1)

# Update the results in the data dictionary
data['test_results'] = {
    'test_0_loss,test_0_accuracy': results_0,
    'test_1_loss,test_1_accuracy': results_1
}

# Write the updated results back to data.txt
with open(f'../../Training/4_split_architecture/{val_acc}/data.txt', 'w') as outfile:
    json.dump(data, outfile)

print("You can find whole details at data.txt on your model folder")
