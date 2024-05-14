from transformers import AutoTokenizer
import torch
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np


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


def average_loss(list_1, list_2):
    if len(list_1) != len(list_2):
        print("Error: The input lists must have the same length.")
        return

    sm = 0
    for i in range(len(list_1)):
        sm += (float(list_1[i]) - float(list_2[i])) ** 2
    print('loss: ' + f'{sm / len(list_1)}')


def plot_classes(g_data, g_time, data, cl, save_path=None):
    plt.figure()  # Create a new figure for each plot
    plt.plot(g_time, g_data, label='generated', color='red', linewidth=2)

    real_data_class_tensors = correct_angels_for_multi_tensor(data[cl], cl)

    i = 0
    for cl_tensors in real_data_class_tensors:
        if i < 10:
            time = pd.DataFrame(cl_tensors['Time'])
            data = pd.DataFrame(cl_tensors[cl])
            plt.plot(time, data, label='Array' + str(i), color='blue', linestyle='--', alpha=0.2)
        i += 1

    plt.xlabel('Time')
    plt.ylabel(cl)
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


def preparing_training_data(data_path, text, class_names):
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




def async_based_on_time(generated_tensors):
    generated_arrays = {label: tensor.detach().numpy() for label, tensor in generated_tensors.items()}
    generated_df = pd.DataFrame(generated_arrays)
    sorted_df = generated_df.sort_values(by='Time')
    sorted_df = sorted_df[sorted_df['Time'] >= 0]
    return sorted_df.reset_index(drop=True)


def create_mean_tiems(texts_tensors):
    time_means = {}
    for text, tensors in texts_tensors.items():
        time_means[text] = {'time': 0, 'number_of_times': 0}
        time = 0
        number_of_times = 0
        ii = 0
        for tensor in tensors:
            time = time + tensor[0]['Time'].iloc[-1]
            number_of_times = number_of_times + len(tensor[0]['Time'])
            ii = ii + 1
        time = time / ii
        time = round(time, 4)
        number_of_times = number_of_times / ii
        number_of_times = round(number_of_times)
        time_means[text]['time'] = time
        time_means[text]['number_of_times'] = number_of_times
    return time_means


def time_number_probabilities(sorted_df, text, mean_time_numbers , percentage=2.5):
    mean_time = mean_time_numbers[text]['time']
    mean_number = mean_time_numbers[text]['number_of_times']
    np.random.seed(2)
    filtered_df = pd.DataFrame()

    for i in range(len(sorted_df)):
        # if (3.14 ** (2 - sorted_df.loc[i, 'Time']/mean_time - i/mean_number) >= percentage):
        if sorted_df.loc[i, 'Time'] < percentage * mean_time:
            filtered_df = pd.concat([filtered_df, sorted_df.iloc[[i]]])
    return filtered_df.reset_index(drop=True)


def add_randomness(data, randomness=0.5, mean=0):
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
def model_tokenizer(words, tokenizer, letngth=19, desired_embed_size=64):
    embeddings_list = []
    for word in words:
        # Tokenize the word and obtain embeddings directly
        embeddings = tokenizer(word, padding='max_length', truncation=True, max_length=letngth, return_tensors='pt')
        embeddings_repeated = embeddings['input_ids'].repeat(desired_embed_size, 1)
        reversed_tensor = embeddings_repeated.T
        embeddings_list.append(reversed_tensor.float())

    stacked_embeddings = torch.stack(embeddings_list)
    return stacked_embeddings

import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers=6, hidden_size=512, num_heads=8, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, src):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        output = self.encoder(src)
        output = self.decoder(output)
        return output

data_name = 'panda_tensor_1'
data_name_pkl = data_name + '.pkl'
removable_columns = ['action', 'who', 'state']
data_path = r'C:\Users\reza\Desktop\transformer\datasets'
data_path = os.path.join(data_path, data_name_pkl)
model_save_path = r"C:\Users\reza\Desktop\transformer\transformer_model.pth"
save_path = r'C:\Users\reza\Desktop\transformer\results'
if not os.path.exists(save_path):
    os.mkdir(save_path)

save_path = os.path.join(save_path, data_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)

X_train, Y_train = get_train_data(data_path, 0.05)
texts_tensors = get_train_data_methdo_2(data_path)
mean_time_numbers = create_mean_tiems(texts_tensors)
class_names = get_classes_names(Y_train)
model = torch.load(model_save_path)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

X = 'b076_Soorati'

tokenized_X_train = model_tokenizer([X], tokenizer)
tokenized_X = tokenized_X_train[0]  # Cast the input tensor to float

save_path = os.path.join(save_path, X)
if not os.path.exists(save_path):
    os.mkdir(save_path)


model = TransformerModel(input_size=64, output_size=64)

# Load the saved parameters into the model
model.load_state_dict(torch.load('transformer_model.pth'))

# Set the model to evaluation mode
model.eval()

output = model(tokenized_X)
# last_state = last_state[0].detach().numpy()
text_training_data_by_classes = preparing_training_data(data_path, X, class_names)

time = output[0][0]
print(save_path)
j = 0
generated_tensors = {}
generated_tensors['Time'] = time
for cl in class_names:
    class_data = output[0][j]
    final_save_path = os.path.join(save_path, cl + 'plot1.png')
    g_data = pd.DataFrame(class_data.detach().numpy())
    g_time = pd.DataFrame(time.detach().numpy())
    # if j == 0:
    plot_classes(g_data, g_time, text_training_data_by_classes, cl, final_save_path)
    generated_tensors[cl] = class_data
    j += 1

sorted_df = async_based_on_time(generated_tensors).reset_index(drop=True)
sorted_df = time_number_probabilities(sorted_df, X, mean_time_numbers, 2.5)
# sorted_df = add_randomness(sorted_df, 0.2)

time = sorted_df['Time']
jj = 0
for cl in class_names:
    if cl != 'Time':
        class_data = sorted_df[cl]
        final_save_path = os.path.join(save_path, cl + 'plot2.png')
        plot_classes(class_data, time, text_training_data_by_classes, cl, final_save_path)
    jj += 1

final_csv_path = os.path.join(save_path, 'data.csv')
sorted_df.to_csv(final_csv_path, index=False, sep='\t')
print(final_csv_path)





