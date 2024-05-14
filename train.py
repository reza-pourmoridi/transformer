from transformers import AutoTokenizer
from transformers import AutoModel
import torch
import os
import numpy as np
import pandas as pd
import pickle
import torch.nn as nn
import torch.optim as optim
import random

data_name = 'panda_tensor'
data_name_pkl = data_name + '.pkl'
removable_columns = ['action', 'who', 'state']
data_path = r'C:\Users\reza\Desktop\transformer\datasets'
data_path = os.path.join(data_path, data_name_pkl)


def turn_vector_to_tensor(vecs, dim_size):
    tensors = []
    for vec in vecs:
        tensor = np.tile(vec, (dim_size, 1)).T
        tensors.append(tensor)
    return tensors


def make_fixed_length(df, desired_length):
    """
    Modifies the DataFrame to have a fixed length by adding or removing rows.

    Args:
        df: The DataFrame to modify.
        desired_length: The desired length of the DataFrame.

    Returns:
        The modified DataFrame with the specified length.
  """

    while df.shape[0] < desired_length:
        # Generate a random index within valid bounds (excluding start and end)
        random_index = np.random.randint(1, df.shape[0])
        # print(random_index)

        # Calculate the average values of the upper and lower index rows
        upper_row = df.iloc[random_index - 1]
        lower_row = df.iloc[random_index]
        average_row = (upper_row + lower_row) / 2

        # Create a new DataFrame with the average values
        # put in the index of random_index
        new_row_df = pd.DataFrame([average_row.values], columns=df.columns)

        # Insert the new row at the specified random index
        df = pd.concat([df.iloc[:random_index], new_row_df, df.iloc[random_index:]])

        # Reset the index to maintain a continuous integer sequence
        df = df.reset_index(drop=True)


    while df.shape[0] > desired_length:
        # Generate a random index within valid bounds (excluding the last index)
        random_index = np.random.randint(0, df.shape[0] - 1)
        # print(random_index)

        # Remove the row at the specified random index
        df = df.drop(random_index)

        # Reset the index to maintain a continuous integer sequence
        df = df.reset_index(drop=True)

    return df


def turn_tensor_to_vector(tensor):
    new_vector = []
    for row in tensor:
        new_vector.append(row.mean())
    return new_vector


def delete_columns(tensor, removable_columns):
    df = pd.DataFrame(tensor)
    for cl in removable_columns:
        if cl in df.columns:
            df = df.drop(cl, axis=1)
    return df


def get_train_data(file_path, percentage=1, removed_itmes=[]):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    robot_actions = []
    tensors = []

    for part_name, part_data in data:
        # print(part_name)
        output_tensor = pd.DataFrame(part_data)
        tensors.append([output_tensor])
        robot_idxs = []
        for i in range(len(part_data)):
            if part_data['who'][i] == 'robot':
                robot_idxs.append(i)
        if robot_idxs:
            first_robot_idx = robot_idxs[0]
            last_robot_idx = robot_idxs[-1]

        # user
        user_idxs = []
        for i in range(len(part_data)):
            if part_data['who'][i] == str(1):
                user_idxs.append(i)
        if user_idxs:
            first_user_idx = user_idxs[0]
            last_user_idx = user_idxs[-1]

        ##### find time padded actions ####################
        # robot
        null_robot_action = ['time_pad'] * (last_robot_idx - first_robot_idx + 1)
        for idx in robot_idxs:
            null_robot_action[idx] = part_data[part_data['who'] == 'robot']['action'][idx]
        robot_actions.append(null_robot_action[0])

    Y = []
    X = []

    i = 0
    for tensor in tensors:
        if i not in removed_itmes:
            Y.append(make_fixed_length(delete_columns(tensor[0], removable_columns), 768))
        i += 1

    i = 0
    for text in robot_actions:
        if i not in removed_itmes:
            X.append(text)
        i += 1

    ten_percent_count = int(len(robot_actions) * percentage)
    X = X[:ten_percent_count]
    Y = Y[:ten_percent_count]

    return X, Y


def process_vectors(tensros, step=0):
    final_values = []
    for tensor in tensros:
        df = pd.DataFrame(tensor)
        df = df.iloc[1:]
        tensor = df.values
        final_values.append([tensor[step]])
    return final_values


def correct_tensors_angels(texts, tensors):
    texts_tensors = {}
    final_texts_tensors = {}
    length_dic = {}
    for i in range(len(texts)):
        texts_tensors[texts[i]] = []
        final_texts_tensors[texts[i]] = []
        length_dic[texts[i]] = []

    for i in range(len(texts)):
        texts_tensors[texts[i]].append(correct_tensor_angels(tensors[i]))
        length_dic[texts[i]].append(len(tensors[i]))

    for tex, tens in texts_tensors.items():
        concatenated_df = pd.concat(tens, ignore_index=True)
        concatenated_df = correct_tensor_angels(concatenated_df)

        sliced_df_list = []
        start = 0
        for length in length_dic[tex]:
            end = start + length
            sliced_df = concatenated_df.iloc[start:end]
            start = end
            final_texts_tensors[tex].append(sliced_df)

    return final_texts_tensors


def correct_tensor_angels(tensor):
    for col in tensor.columns:
        tensor = correct_angels(tensor, col)
    return tensor


def correct_angels(angels, class_name):
    diff_f = False
    for i in range(1, len(angels)):
        # Calculate the absolute difference between consecutive H_A1 values
        diff = abs(angels.loc[i, class_name] - angels.loc[i - 1, class_name])

        # If the difference is greater than 180, subtract 10 from the current H_A1 value
        if diff > 180:
            diff_f = True
            if (angels.loc[i, class_name] > 180):
                angels.loc[i, class_name] -= 360

    if diff_f:
        angels = correct_angels_reverse(angels, class_name)
    return angels


def correct_angels_reverse(angels, class_name):
    diff_f = False
    for i in range(len(angels) - 1, -1, -1):  # Iterating in reverse order, including index 0
        # Calculate the absolute difference between consecutive H_A1 values
        diff = 0
        if i + 2 < len(angels):
            diff = abs(angels.loc[i, class_name] - angels.loc[i + 1, class_name])

        # If the difference is greater than 180, subtract 10 from the current H_A1 value
        if diff > 180:
            diff_f = True
            if angels.loc[i, class_name] > 180:
                angels.loc[i, class_name] -= 360
    return angels


def process_data(input_outputs_dic):
    # probabilities_based_time_stpes = {}
    # for text, tensors in input_outputs_dic.items():
    #     probabilities_based_time_stpes[text] =
    # print(input_outputs_dic['b078_sabz'])
    # raise Exception

    return 1, 2


def probabilities_parameters(texts_tensors):
    time_means = {}
    for text, tensors in texts_tensors.items():
        time_means[text] = {'time': 0, 'number_of_times': 0}
        time = 0
        number_of_times = 0
        ii = 0
        for tensor in tensors:
            time = time + tensor['Time'].iloc[-1]
            number_of_times = number_of_times + len(tensor['Time'])
            ii = ii + 1
        time = time / ii
        time = round(time, 4)
        number_of_times = number_of_times / ii
        number_of_times = round(number_of_times)
        time_means[text]['time'] = time
        time_means[text]['number_of_times'] = number_of_times
    return time_means


def process_tensors(input_outputs_dic, rows=768):
    expanded_tensors = []
    texts = []
    for text, tensors in input_outputs_dic.items():
        for tensor in tensors:
            df = tensor.values
            df = pd.DataFrame(df)
            df = df.to_numpy()
            padding_rows = rows - df.shape[0]
            padding_array = np.zeros((padding_rows, df.shape[1]))
            expanded_tensor = np.concatenate((df, padding_array), axis=0)
            expanded_panda_tensor = pd.DataFrame(expanded_tensor)
            expanded_panda_tensor = expanded_panda_tensor.T
            texts.append(text)
            expanded_tensors.append(expanded_panda_tensor)
    texts, expanded_tensors = shuffle_synced_lists(texts, expanded_tensors)
    return texts, expanded_tensors


def shuffle_synced_lists(list1, list2):
    paired_lists = list(zip(list1, list2))
    random.shuffle(paired_lists)
    shuffled_list1, shuffled_list2 = zip(*paired_lists)

    return list(shuffled_list1), list(shuffled_list2)


def add_randomness(data, randomness=1, mean=0):
    data = data.astype(float)  # Cast the DataFrame to float dtype
    for index, row in data.iterrows():
        for col in data.columns:
            if col == "Time":
                original_value = float(row[col])  # Convert original value to float
                data.at[index, col] = float('{:.4f}'.format(original_value))
            else:
                original_value = float(row[col])  # Convert original value to float
                rand_value = np.random.normal(mean, randomness)  # Generate randomness from normal distribution
                data.at[index, col] = float('{:.4f}'.format(original_value + rand_value))
    return data


shure_delete = [22 ,35,50,51,55,72]
relative_delete = [9,14,29,30,35,53,58,83,97,110,127,134]
removed_items = shure_delete + relative_delete
removed_items = []

X_train, Y_train = get_train_data(data_path, 1, removed_items)

input_outputs_dic = correct_tensors_angels(X_train, Y_train)
probabilities_parameters = probabilities_parameters(input_outputs_dic)

num_epochs = 15
batch_size = 64
learning_rate = 0.001
model_save_path = r"C:\Users\reza\Desktop\transformer\results\model.pt"

model = AutoModel.from_pretrained("albert-base-v1")
num_params = model.num_parameters()
print("Number of parameters in the model:", num_params)

tokenizer = AutoTokenizer.from_pretrained("albert-base-v1")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    print('---------------------------------------------------------------')
    print('training epochs ' + f'{epoch + 1}' + ' batch_size= ' f'{batch_size}')
    print()
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i + batch_size]
        batch_y_tensors = Y_train[i:i + batch_size]
        numpy_array = np.stack(batch_y_tensors)
        batch_y_torch_tensor = torch.from_numpy(numpy_array).float()  # Convert to float32

        tokenized_X_train = [tokenizer(word, padding='max_length', truncation=True, max_length=19, return_tensors='pt')
                             for word in batch_X]
        input_X_ids = torch.cat([tokenized_text['input_ids'] for tokenized_text in tokenized_X_train], dim=0)
        attention_mask_X = torch.cat([tokenized_text['attention_mask'] for tokenized_text in tokenized_X_train], dim=0)
        token_type_ids_X = torch.cat([tokenized_text['token_type_ids'] for tokenized_text in tokenized_X_train], dim=0)

        optimizer.zero_grad()
        output = model(input_ids=input_X_ids, attention_mask=attention_mask_X, token_type_ids=token_type_ids_X)
        last_state = output.last_hidden_state
        loss = criterion(last_state, batch_y_torch_tensor)
        print('loss:' f'{loss}')

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(X_train)}], Loss: {loss.item()}')

torch.save(model, model_save_path)
