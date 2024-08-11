# Function to split dataset
def split_dataset(dataset, train_ratio=0.45, val_ratio=0.45, test_ratio=0.10):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Get all file names
    all_files = [os.path.basename(dataset.imgs[i][0]) for i in range(total_size)]

    # Sort file names for reproducibility
    all_files.sort()

    # Use a fixed random seed for reproducibility
    np.random.seed(80)

    # Shuffle the sorted file names
    np.random.shuffle(all_files)

    # Split the shuffled file names
    train_files = set(all_files[:train_size])
    val_files = set(all_files[train_size:train_size+val_size])
    test_files = set(all_files[train_size+val_size:])

    # Create indices for each split
    train_indices = [i for i in range(total_size) if os.path.basename(dataset.imgs[i][0]) in train_files]
    val_indices = [i for i in range(total_size) if os.path.basename(dataset.imgs[i][0]) in val_files]
    test_indices = [i for i in range(total_size) if os.path.basename(dataset.imgs[i][0]) in test_files]

    # Create subsets
    trainset = Subset(dataset, train_indices)
    valset = Subset(dataset, val_indices)
    testset = Subset(dataset, test_indices)

    return trainset, valset, testset

# Split the dataset
trainset, valset, testset = split_dataset(dataset)

print(f"Train set size: {len(trainset)}")
print(f"Validation set size: {len(valset)}")
print(f"Test set size: {len(testset)}")
