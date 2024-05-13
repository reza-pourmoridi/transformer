import torch
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np



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



def delete_columns(tensor, removable_columns):
    df = pd.DataFrame(tensor)
    for cl in removable_columns:
        if cl in df.columns:
            df = df.drop(cl, axis=1)
    return df

def get_train_data(file_path, percentage=1):
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

def plot_classes(data):
    plt.figure()  # Create a new figure for each plot
    time = pd.DataFrame(data['Time'])

    for col in data.columns:
        if col != 'Time':
            col_data = pd.DataFrame(data[col])
            col_data = correct_angels(col_data, col)
            plt.plot(time, col_data, label=str(col), color='blue', linestyle='--', alpha=0.2)


    plt.xlabel('Time')
    plt.ylabel('columns')
    plt.show()


data_name = 'c1_data'
data_name_pkl = data_name + '.pkl'
removable_columns = ['action', 'who', 'state']

data_path = r'C:\Users\reza\Desktop\transformer\datasets'
data_path = os.path.join(data_path, data_name_pkl)
X_train, Y_train = get_train_data(data_path, 1)

i = 1
for tensor in Y_train:
    print(i)
    plot_classes(tensor)
    i += 1


shure_delete = [22 ,35,50,51,55,72]
relative_delete = [9,14,29,30,35,53,58,83,97,110,127,134]







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