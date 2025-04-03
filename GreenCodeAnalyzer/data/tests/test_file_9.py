import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

def remove_files_based_on_sequence(root_dir):

    counter = 0
    # If row of file has number of sequence numbers less than the directory it is stored under, delete the file
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            # Get the number of sequence numbers from the directory name
            persistence = root_dir.split('_')[2].split('a')[0]
            seq_nums = []
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)

                # Check seq number of each row and if not added yet, then add to the list
                for row in reader:
                    try:
                        seq_num = int(float(row[-1]))
                    except ValueError:
                        print(f"Error converting sequence number from row: {row}")
                        continue
                    if seq_num not in seq_nums:
                        seq_nums.append(seq_num)

            if len(seq_nums) < int(persistence):
                os.remove(file_path)
                counter += 1

    print(f"Deleted {counter} files.")


def plot_movement(csv_file_path):
    df_col_list = ['range', 'azim', 'dopp', 'snr', 'y', 'x', 'current_frame', 'seq']
    # Load the CSV file
    df = pd.read_csv(csv_file_path, names=df_col_list)

    # Define the number of frames to plot
    frame_numbers = df['seq'].unique()

    # Create a single plot for all frames
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define color map
    colors = [(0, 0.447, 0.741), (0.85, 0.325, 0.098),  (0.3, 0.8, 1.0), (0.466, 0.674, 0.188)]

    # Counter for colors
    counter = 0

    # Iterate over each frame
    for frames in frame_numbers:
        subset = df[df['seq'] == frames]
        frame_data = subset.groupby('current_frame')

        # Plot each frame with a different color
        for frame, data in frame_data:
            ax.scatter(data['x'], data['y'], label=f'Frame {frame}', color=colors[counter % len(colors)], alpha=0.5, s=200)

        counter += 1

    # Create custom legend
    handles, labels = ax.get_legend_handles_labels()
    large_dots = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                             markerfacecolor=handle.get_facecolor()[0], markersize=15)
                  for handle, label in zip(handles, labels)]
    ax.legend(handles=large_dots, loc='best')

    # Make font size of axes labels larger
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlim([-8, 0])
    ax.set_ylim([4, 8])
    ax.set_xlabel('X', fontsize=17)
    ax.set_ylabel('Y', fontsize=17)
    ax.set_title('Movement Across Frames', fontsize=24)

    plt.tight_layout()
    plt.show()


def count_files(root_dir):
    category_counts = {}
    total_counts = {'train': 0, 'valid': 0, 'test': 0, 'total': 0}

    # Loop through each category in the root directory
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)

        if os.path.isdir(category_path):
            # Initialize counts for train, validation, and test
            counts = {'train': 0, 'valid': 0, 'test': 0}

            for subdir in counts.keys():
                subdir_path = os.path.join(category_path, subdir)

                if os.path.exists(subdir_path):
                    # Count the number of files in the subdirectory
                    file_count = len(
                        [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])
                    counts[subdir] = file_count
                    total_counts[subdir] += file_count

            category_counts[category] = counts
            total_counts['total'] += sum(counts.values())

    return category_counts, total_counts


if __name__ == "__main__":
    csv_file_path = input("Enter the root directory path: ")
    plot_movement(csv_file_path)
