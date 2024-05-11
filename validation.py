import torch
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer
from transformers import AutoModel
import torch
import os
import numpy as np
import pandas as pd
import pickle
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


data_name = 'c1_data'
text = 'b078_sabz'
data_name_pkl = data_name + '.pkl'
removable_columns = ['action', 'who', 'state']
model_save_path = r"C:\Users\reza\Desktop\transformer\model.pt"
data_path = r'C:\Users\reza\Desktop\transformer\datasets'
data_path = os.path.join(data_path, data_name_pkl)


save_path = r'C:\Users\reza\Desktop\transformer\results'
text = 'c1_data'
if not os.path.exists(save_path):
    os.mkdir(save_path)

save_path = os.path.join(save_path, data_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)


def turn_vector_to_tensor(vecs, dim_size):
    tensors = []
    for vec in vecs:
        tensor = np.tile(vec, (dim_size, 1)).T
        tensors.append(tensor)
    return tensors

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


def get_train_data(file_path, percentage = 1):
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
    X = robot_actions
    for tensor in tensors:
        Y.append(delete_columns(tensor[0], removable_columns))

    ten_percent_count = int(len(robot_actions) * percentage)
    X = X[:ten_percent_count]
    Y = Y[:ten_percent_count]

    return X, Y

def get_train_data_methdo_2(file_path):
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

    # create a dic that contains each text with its tensors.
    texts_tensors = {}
    for text in robot_actions:
        texts_tensors[text] = []

    i = 0
    for text in robot_actions:
        texts_tensors[text].append(tensors[i])
        i = i + 1

    return texts_tensors

def process_tensors(tensros, step = 0):
    final_values = []
    for tensor in tensros:
        df = pd.DataFrame(tensor)
        df = df.iloc[1:]
        tensor = df.values
        final_values.append([tensor[step]])
    return final_values


def average_loss(list_1, list_2):
    if len(list_1) != len(list_2):
        print("Error: The input lists must have the same length.")
        return

    sm = 0
    for i in range(len(list_1)):
        sm += (float(list_1[i]) - float(list_2[i])) ** 2
    print('loss: ' + f'{sm / len(list_1)}')


def plot_classes(g_data, g_time, data, cl, save_path=None):
    save_path = os.path.join(save_path, cl + 'plot.png')
    plt.figure()  # Create a new figure for each plot
    g_data = pd.DataFrame(g_data.detach().numpy())
    g_time = pd.DataFrame(g_time.detach().numpy())
    plt.plot(g_time, g_data, label='generated', color='red', linewidth=2)

    real_data_class_tensors = correct_angels_for_multi_tensor(data[cl], cl)
    colors = ['#0000ff', '#0c0cff', '#1919ff', '#2626ff', '#3232ff', '#3f3fff',
              '#4c4cff', '#5959ff', '#6565ff', '#7272ff', '#7f7fff', '#8b8bff',
              '#9898ff', '#a5a5ff', '#b1b1ff', '#bebeff', '#cbcbff', '#d8d8ff', '#d8d8ff', '#d8d8ff']


    i = 0
    for cl_tensors in real_data_class_tensors:
        if i < 5:
            data = pd.DataFrame(cl_tensors)
            time = pd.DataFrame(cl_tensors['Time'])
            plt.plot(time, data, label='Array' + str(i), color='blue', linestyle='--', alpha=0.5)
        i += 1

    plt.xlabel('Time')
    plt.ylabel(cl)
    plt.title('generated')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def correct_angels_for_multi_tensor(data, class_name):
    new_data_list = []
    length_list = []
    for cl_tensors in data:
        angels = correct_angels(cl_tensors[0], class_name)
        new_data_list.append(angels)
    for list in new_data_list:
        length_list.append(len(list))

    concatenated_df = pd.concat(new_data_list, ignore_index=True)
    concatenated_df = correct_angels(concatenated_df, class_name)

    sliced_df_list = []
    start = 0
    for length in length_list:
        end = start + length
        sliced_df = concatenated_df.iloc[start:end]
        sliced_df_list.append(sliced_df)
        start = end

    return sliced_df_list


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




def get_classes_names(tensors):
    labels = tensors[0].columns
    labels_except_time = [label for label in labels if label != 'Time']
    return labels_except_time


def preparing_training_data(data_name, text, class_names):
    data_path = r'C:\Users\reza\Desktop\transformer\datasets'
    data_name_pkl = data_name + '.pkl'
    data_path = os.path.join(data_path, data_name_pkl)
    train_tensor = get_train_data_methdo_2(data_path)

    text_tensors = train_tensor[text]
    time_joined_tensors = {}
    for tensor in text_tensors:
        tensor = tensor[0]
        for cl in class_names:
            time_joined_tensors[cl] = []

    for tensor in text_tensors:
        tensor = tensor[0]
        for cl in class_names:
            time_joined_tensors[cl].append([pd.concat([tensor['Time'], tensor[cl]], axis=1)])

    return time_joined_tensors

X_train, Y_train = get_train_data(data_path , 0.1)
class_names = get_classes_names(Y_train)
Y_train = process_tensors(Y_train, 0)
Y_train_tensors = turn_vector_to_tensor(Y_train, 768)



model = torch.load(model_save_path)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")



for i in range(len(X_train)):
    X = X_train[i]
    Y = Y_train_tensors[i]

    tokenized_X = tokenizer(X, padding='max_length', truncation=True, max_length=19, return_tensors='pt')

    final_save_path = os.path.join(save_path, X)
    if not os.path.exists(final_save_path):
        os.mkdir(final_save_path)

    output = model(**tokenized_X)
    last_state = output.last_hidden_state
    # last_state = last_state[0].detach().numpy()
    text_training_data_by_classes = preparing_training_data(data_name, X, class_names)
    j = 0

    print(final_save_path)
    for cl in class_names:
        time = last_state[0][0]
        class_data = last_state[0][j]
        plot_classes(class_data,time ,text_training_data_by_classes, cl, final_save_path)
        j += 1





