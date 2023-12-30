def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = (xB - xA) * (yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou if iou > 0 and iou <= 1  else 0

def create_directory(path):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)
def to_xyxy(xywh):
    x, y, w, h = xywh
    return [x-(w/2),y-(h/2),x+(w/2),y+(h/2)]

def hamming(str1, str2):
    i = count = 0
    while(i < len(str1)):
        count += 1 if(str1[i] != str2[i]) else 0
        i += 1
    #print("Hamming(",str1,",",str2,") =", count)
    return count

def print_divider():
    print("\n============================================================\n")

def frequency(list, percentage=False):
    frequency = {}
    for item in list:
        if item in frequency: frequency[item] += 1
        else: frequency[item] = 1
    total = sum(frequency.values())
    pfrequency = {x : frequency[x]/total for x in frequency}
    return pfrequency if percentage else frequency

def normalize_utc(code):
    from .config import UTC_START_TIME
    
    return (int(code) - UTC_START_TIME)/100

def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)

import numpy as np
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def sensor_update_matrix(data, sensor_path):
    import pandas as pd
    from .Matrix import Matrix
    import time
    sensor_df = pd.read_csv(sensor_path)
    sensor_df["time"] = sensor_df["time"] / 1000000
    sensor_times = sensor_df["time"].values.tolist()
    for label in data:
        start = time.time()
        sensor_time = find_nearest(sensor_times, value=int(label.utc))
        sensor_row = sensor_df[sensor_df['time']==sensor_time]
        sensor_pitch = sensor_row['pitch'].values.tolist()[0]
        sensor_yaw   =   sensor_row['yaw'].values.tolist()[0]
        sensor_roll  =  sensor_row['roll'].values.tolist()[0]
        label.matrix = Matrix(yaw=sensor_yaw, pitch=sensor_pitch, roll=sensor_roll )
        end = time.time()
        print("Time elapsed:", end-start)
    print("Update Completed")
    return data

def string_sequence(sequence):
    return "".join(str(x[0]) for x in sequence)
    
def number_sequence(sequence):
    return [x[0] for x in sequence]

def print_sequences(sequence_list):
    for s in sequence_list:
        print_divider()
        print(string_sequence(s))
        #print([x[1] for x in s])
    print_divider()

# Takes 2 number sequences of same lengas input 
def similarity(s1, s2):
    intersection = sum([1 if int(s2[i]) > 0 and int(s1[i]) > 0 else 0 for i in range(len(s1))])
    union = sum([1 if int(s1[i]) > 0 or int(s2[i]) > 0 else 0 for i in range(len(s1))])
    return [0 if union == 0 else intersection/union , intersection, union]

def calculate_gamma(B_V, B_C, tau):
    # Ensure the inputs are numpy arrays
    B_V = np.array(B_V)
    B_C = np.array(B_C)
    if B_V.shape != B_C.shape:
        raise ValueError("The shapes of B_V and B_C must be the same.")

    # Calculating the numerator (number of positions where B_V(i, j) equals B_C(i, j) and B_V(i, j) is not 0)
    numerator = np.sum((B_V == B_C) & (B_V != 0))

    # Calculating the denominator (number of positions where either B_V(i, j) or B_C(i, j) is not 0)
    denominator = np.sum((B_V != 0) | (B_C != 0))

    # Calculating gamma, avoiding division by zero
    gamma = 0
    if denominator != 0:
        gamma = int((numerator / denominator) >= tau)
    else:
        return -1
    return gamma

import matplotlib.pyplot as plt
def df_to_table(df, save=False, show=False, filename="None"):
    dir = "plots/save/tables/"
    create_directory(dir)
    plt.axis('off')
    plt.table(cellText=df.values, colLabels=df.columns, loc='center')

    if save: plt.savefig(dir+filename+".png")
    if show: plt.show()
    plt.clf()

from utils.config import COLS
def df_to_plot(df, x, ys, xlabel="",ylabel="",save=False, show=False, distance_sweep=False, filename="None", title="", colors=("blue", "red"), axs=None, label=None):
    dir = "plots/results/"
    create_directory(dir)
    plt.axis('on')
    colors = {
                "Passing Rate" : colors[0], 
                "Adversary Passing Rate" : colors[1],
                "E(VC)" : colors[0],
                "E(A)" : colors[1]
            }
    # add dotted for adv
    x = x[0] if distance_sweep else x
    cols = COLS
    for plot in ys:
        index = plot[0]
        categories = plot[1]
        #label = plot[2]
        for category in categories:
            is_adversary = category in ["Adversary Passing Rate","E(A)"]
            linestyle = "--" if is_adversary else "-"
            temp="A" if is_adversary else "C"#label if not is_adversary else None
            # color = colors[category]
            color = "Red" if is_adversary else "Blue"
            try:
                axs[index//cols][index%cols].plot(df[x], df[category], color=color, label=temp, linestyle=linestyle)
                axs[index//cols][index%cols].set_title(title)
                axs[index//cols][index%cols].grid(True)
            except:
                axs.plot(df[x], df[category], color=color, label=temp, linestyle=linestyle)
                axs.set_title(title)
                axs.grid(True)
    plt.legend()
    if save: 
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(dir+filename+".png")
        plt.close()
    if show: plt.show()
    # plt.clf()

def convert_table_to_plot(table, distance_sweep=False, filename="None", xlab="None", ylab="Passing Rate"):

    dir = "plots/temp/"
    create_directory(dir)
    x = []
    y = []
    z = []
    total = []
    for t in table[1:-1]:
        x.append(t[0])
        y.append(round(float(t[3])* 100, 2))
        z.append(round(float(t[4])* 100, 2))
        total.append(t[1]+t[2])

    x = [a[0] for a in x] if distance_sweep else x
    plt.plot(x,y,color="blue")
    plt.plot(x,z,color="red")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.savefig(dir+filename)
    plt.clf()

def table_to_csv(data, filename="None"):
    import pandas as pd
    dir = "plots/results/"
    create_directory(dir)
    df = pd.DataFrame(data)
    df.to_csv(dir+filename+".csv")

def integer_to_binary(n, bits):
    # Convert integer to binary string and remove the '0b' prefix
    binary_str = bin(n)[2:]
    
    # Pad the binary string with leading zeros to ensure it has the specified bit length
    padded_binary_str = binary_str.zfill(bits)
    
    # Truncate the binary string to the desired number of bits if it exceeds the specified length
    truncated_binary_str = padded_binary_str[-bits:]
    
    return truncated_binary_str

def binary(num_regions):
    # Calculate the maximum value in the interval [0, (2^num_regions) - 1]
    max_val = 2 ** num_regions - 1
    
    # Construct the dictionary mapping each integer in the interval to its binary representation
    mapping = {i: integer_to_binary(i, num_regions) for i in range(max_val + 1)}
    return mapping



def split_dataframe(df):
    # Dictionary to store the split dataframes
    split_dfs = {}

    # Iterate over each unique combination of N, k, and Alpha
    for n, k, alpha in df[['K', 'k', 'alpha']].drop_duplicates().to_numpy():
        # Filter the dataframe for the current combination
        filtered_df = df[(df['K'] == n) & (df['k'] == k) & (df['alpha'] == alpha)]

        # Add the filtered dataframe to the dictionary
        split_dfs[(n, k, alpha)] = filtered_df

    return split_dfs

def process_run(base_directory, threshold=False):
    import os
    import pandas as pd
    root = 'plots/results/csv/'
    # Directory containing the CSV files
    base = base_directory
    directory = os.path.join(root, base)

    # List to hold data from each CSV file
    dataframes = []
    
    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    if "Unnamed: 0" in combined_df.columns.to_list():
        print(combined_df.columns)
        print("DELETING UNNAMED COLUMN")
        combined_df = combined_df.drop(columns = ["Unnamed: 0"])
        print(combined_df.columns)
    path = root + "processed/data-" + base + ".csv"
    combined_df.to_csv(path)
    process_plots(path, base, threshold=threshold)



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
def reorganize_dict(original_dict):
    # Create a new dictionary with keys as the first elements of the tuples
    new_dict = {}
    # Iterate over the items in the original dictionary
    for key, value in original_dict.items():
        # Extract the first element of the tuple (key)
        N, k, a = key
        # Initialize the list for this key if not already done
        if (N,k) not in new_dict:
            new_dict[(N,k)] = {}
        # Append the current tuple and its value to the list for this key
        new_dict[(N,k)][a] = value
    return new_dict

def resample_data(x, y, resolution):
    x_resampled = np.arange(x.min(), x.max(), resolution)
    y_resampled = np.interp(x_resampled, x, y)
    return x_resampled, y_resampled

def moving_average(data, window_size, step):
    return data.rolling(window=window_size, min_periods=1).mean()[::step]

def process_plots(file_path, base, resolution=0.05, window_length=1, stride=1,  threshold=False):
    results = pd.read_csv(file_path)
    dfs = reorganize_dict(split_dataframe(results))
    color_map = {}
    for (N, k), alpha_data in dfs.items():
        plt.figure(figsize=(10, 6))
        color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        for column, linestyle in [("Passing Rate", "-"), ("Adversary Passing Rate", "--")]:
            
            for alpha in [x for x in alpha_data if x>0.29]:
                data = alpha_data[alpha]
                x, y = round(data['beta']*N), data[column] / 100
                x_resampled, y_resampled = resample_data(x, y, resolution)
                y_avg = moving_average(pd.Series(y_resampled), window_length, stride)
                x_avg = x_resampled[:len(y_avg)]
                # x_avg = x
                # y_avg = y
                # Assign a color to each (N, k, alpha) tuple if not already assigned
                if (N, k, alpha) not in color_map:
                    color_map[(N, k, alpha)] = next(color_cycle)
                color = color_map[(N, k, alpha)]
                if not threshold:
                    plt.plot(x_avg, y_avg, label=f'α={round(alpha*k)}/{round(k)}' if linestyle != "--" else None, color=color, linestyle=linestyle)
        if threshold:
            plt.plot(results['alpha'], results['Passing Rate'], label='Passing Rate', marker='o')
            plt.plot(results['alpha'], results['Adversary Passing Rate'], label='Adversary Passing Rate', linestyle='dotted', marker='x')
    
        plt.title(f'Passing Rates for K={int(N)}, k={int(k)}')
        plt.xlabel('Beta' if not threshold else 'Alpha')
        plt.ylabel('Rate')
        plt.legend(title='Solid: CPR, Dotted: APR')
        plt.grid(True)
        dir = f"plots/results/plots/{base}/"
        create_directory(dir)
        plt.savefig(f"{dir}/K={int(N)}_k={int(k)}.png")

def plot_threshold(df):
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['alpha'], df['Passing Rate'], label='Passing Rate', marker='o')
    plt.plot(df['alpha'], df['Adversary Passing Rate'], label='Adversary Passing Rate', linestyle='dotted', marker='x')
    plt.xlabel('Alpha')
    plt.ylabel('Rate')
    plt.title('Passing Rate and Adversary Passing Rate vs Alpha')
    plt.legend()
    plt.grid(True)
    plt.show()
