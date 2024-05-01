import numpy as np
from keras.utils import to_categorical


def spliting_input(input_array):
    # Each element in input_array is expected to be a numpy ndarray
    input_parts = [[], [], [], []]

    for sample in input_array:
        total_samples = len(sample)
        part_size = total_samples // 4

        for i in range(4):
            start = i * part_size
            if i == 3:  # Handle last part possibly containing more elements
                end = total_samples
            else:
                end = start + part_size
            part_sample = sample[start:end]
            swap_axis02 = np.swapaxes(part_sample, 0, 2)
            swap_axis_final = np.swapaxes(swap_axis02, 0, 1)
            input_parts[i].append(swap_axis_final)

    # Convert lists to arrays before returning
    return [np.array(part) for part in input_parts]


def name_it_later(examples_evaluation, labels_evaluation):
    extended_evaluation_set = [[], [], [], []]
    extended_label_set = []

    for examples, labels in zip(examples_evaluation, labels_evaluation):
        for example, label in zip(examples, labels):
            for part in range(4):
                extended_evaluation_set[part].extend(example[part])
            extended_label_set.extend(label)

    # Make sure that each part in extended_evaluation_set is a numpy array
    extended_evaluation_set = [np.array(part) for part in extended_evaluation_set]

    # Split the inputs into four parts
    split_inputs = spliting_input(extended_evaluation_set)

    # Encode the labels
    evaluation_encoded_labels = to_categorical(np.array(extended_label_set))

    X = split_inputs
    Y = evaluation_encoded_labels
    return X, Y


# Load dataset
examples_test_0, labels_test_0 = np.load("../NPY_files/loaded_Test_0.npy", allow_pickle=True)
examples_test_1, labels_test_1 = np.load("../NPY_files/loaded_Test_1.npy", allow_pickle=True)

# Process and save datasets
test_0 = name_it_later(examples_test_0, labels_test_0)
np.save("NPY_files/prepared_test0_dataset.npy", test_0)

test_1 = name_it_later(examples_test_1, labels_test_1)
np.save("NPY_files/prepared_test1_dataset.npy", test_1)

print("Done, it's time to evaluate your model.")
print("Run evaluate_model.py, make sure you assign the model folder name to val_acc on evaluate_model.py before run")

# Print shapes of the test datasets
print("Shapes of test_0 inputs:")
for part in test_0[0]:
    print(part.shape)
print("Shape of test_0 labels:", test_0[1].shape)

print("Shapes of test_1 inputs:")
for part in test_1[0]:
    print(part.shape)
print("Shape of test_1 labels:", test_1[1].shape)