import pandas as pd
import numpy as np
import joblib
import os
import json
import pickle
import matplotlib.pyplot as plt
import warnings
import shutil
import random


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
            Y.append(delete_columns(tensor[0], removable_columns))
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


def delete_columns(tensor, removable_columns):
    df = pd.DataFrame(tensor)
    for cl in removable_columns:
        if cl in df.columns:
            df = df.drop(cl, axis=1)
    return df



def predict_class_numbers(project_path, data_name, text, class_name, new_time):
    model_path = os.path.join(project_path, data_name, text, class_name + '_model.joblib')
    model = joblib.load(model_path)
    new_time_reshaped = new_time.reshape(-1, 1)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    predicted_class_nums = model.predict(new_time_reshaped)
    return predicted_class_nums


def compare_with_train_data(project_path, data_name, text, class_name):
    data_path = r'C:\Users\reza\Desktop\transformer\datasets'
    data_name_pkl = data_name + '.pkl'
    data_path = os.path.join(data_path, data_name_pkl)
    train_tensor = get_train_data(data_path)
    class_tensor = train_tensor[text][class_name]


def preparing_training_data(project_path, data_name, text, class_names):
    data_path = r'C:\Users\reza\Desktop\transformer\datasets'
    data_name_pkl = data_name + '.pkl'
    data_path = os.path.join(data_path, data_name_pkl)
    train_tensor = get_train_data(data_path)
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


def plot_classes(g_data, g_time, data, cl, save_path=None):
    plt.figure()  # Create a new figure for each plot

    g_data = pd.DataFrame(g_data)
    g_time = pd.DataFrame(g_time)
    plt.plot(g_time, g_data, label='generated', color='red', linewidth=2)

    real_data_class_tensors = correct_angels_for_multi_tensor(data[cl], cl)
    colors = ['#0000ff', '#0c0cff', '#1919ff', '#2626ff', '#3232ff', '#3f3fff',
              '#4c4cff', '#5959ff', '#6565ff', '#7272ff', '#7f7fff', '#8b8bff',
              '#9898ff', '#a5a5ff', '#b1b1ff', '#bebeff', '#cbcbff', '#d8d8ff', '#d8d8ff', '#d8d8ff']


    i = 0
    for cl_tensors in real_data_class_tensors:
        data = pd.DataFrame(cl_tensors)
        time = pd.DataFrame(cl_tensors['Time'])
        plt.plot(time, data, label='Array' + str(i), color=colors[i])
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
    concatenated_df = correct_angels(concatenated_df, cl)

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



def generate_sin_data(times):
    phase_shift = random.uniform(-1, 1)  # Random phase shift
    amplitude_scale = random.uniform(3, 15)  # Random amplitude scaling
    sine_wave = 100 + np.sin(times + phase_shift) * amplitude_scale
    return sine_wave

# Load the trained model from the saved file

data_name = 'c1_data'
# text = 'b077_Narenji'

# data_name = input('enter the dataset name: ')
# text = input('enter the text you want for generating: ')
data_name = 'c1_data'
data_name_pkl = data_name + '.pkl'
removable_columns = ['action', 'who', 'state']
data_path = r'C:\Users\reza\Desktop\transformer\datasets'
data_path = os.path.join(data_path, data_name_pkl)

project_path = r'C:\Users\reza\Desktop\transfer_model\train\data_names'
save_path = r'C:\Users\reza\Desktop\transformer\test_dataset'
if not os.path.exists(save_path):
    os.mkdir(save_path)

save_path = os.path.join(save_path, data_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)


def generate_pd_data(data_name, text):
    main_config_json_path = os.path.join(project_path, data_name, 'main_config.json')
    with open(main_config_json_path, 'r') as main_config_json_file:
        class_names = json.load(main_config_json_file)

    json_path = os.path.join(project_path, data_name, text,'config.json')
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

    mean = json_data['time'] + 2
    std_dev = 3
    time_noise = abs(np.random.normal(mean, std_dev))
    time = json_data['time'] + time_noise

    mean = json_data['number_of_times'] + 15
    std_dev = 20
    number_of_times_noise = np.random.normal(mean, std_dev)
    number_of_times = round(json_data['number_of_times'] + number_of_times_noise)
    new_time = np.linspace(0, time, number_of_times)
    who_tensor = pd.Series(['user'] * number_of_times)
    action_tensor = pd.Series(['time_pad'] * number_of_times)
    action_tensor[0] = text
    who_tensor[0] = 'robot'


    generated_tensors = {}
    generated_tensors['Time'] = new_time
    generated_tensors['who'] = who_tensor
    generated_tensors['action'] = action_tensor


    for cl in class_names:
        generated_tensors[cl] = generate_sin_data(generated_tensors['Time'])


    generated_tensors = pd.DataFrame(generated_tensors)

    return generated_tensors


# generated_tensors = generate_pd_data(data_name, text)

# final_csv_path = os.path.join(save_path, 'data.csv')
# generated_tensors.to_csv(final_csv_path, index=False, sep='\t')
# print(final_csv_path)

removed_items = []

X_train, Y_train = get_train_data(data_path, 1, removed_items)
Y_train = []

times = 100
for i in range(times):
    for X in X_train:
        generated_tensor = generate_pd_data(data_name, X)
        final = (X, generated_tensor)
        Y_train.append(final)

with open('datasets/panda_tensor_100.pkl', 'wb') as f:
    pickle.dump(Y_train, f)




file_path = r'C:\Users\reza\Desktop\transformer\datasets\panda_tensor_100.pkl'

x,y = get_train_data(file_path)
print(len(x))


