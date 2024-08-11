# Define the URLs
dataset_url = 'https://drive.google.com/uc?id=1V0c8p6MOlSFY5R-Hu9LxYZYLXd8B8j9q'
labels_url = 'https://raw.githubusercontent.com/cleardusk/MeGlass/master/meta.txt'

# Define file names and |
dataset_zip = 'MeGlass_120x120.zip'
labels_file = 'meta.txt'
base_dir = 'MeGlass'
glasses_dir = os.path.join(base_dir, 'glasses')
no_glasses_dir = os.path.join(base_dir, 'no_glasses')

# Download the dataset if not yet downloaded
if not os.path.exists(dataset_zip):
    print(f"Zip file {dataset_zip} not found. Downloading...")
    gdown.download(dataset_url, dataset_zip, quiet=False)
else:
    print(f"Dataset zip file {dataset_zip} already exists. Skipping download.")

# Download (if not already) the labels file
!if test -f meta.txt; then \
    echo "Labels file already exists."; \
else \
    wget -O meta.txt https://raw.githubusercontent.com/cleardusk/MeGlass/master/meta.txt && \
    echo "Labels file downloaded."; \
fi

# Check the labels
with open('meta.txt', 'r') as f:
    lines = f.readlines()
print("Number of labels:", len(lines))

# Unzip and the dataset
!unzip -j -q MeGlass_120x120.zip -d MeGlass

# Print size of MeGlass dataset
print("Size of MeGlass:", len(os.listdir('MeGlass')))

# Create directories for glasses and no glasses if they don't exist
os.makedirs(glasses_dir, exist_ok=True)
os.makedirs(no_glasses_dir, exist_ok=True)

# Split the dataset based on the labels
for line in lines:
    # Split the line into filename and label
    filename, label = line.strip().split()
    src_path = os.path.join(base_dir, filename)
    # Deliver to correct folder
    if label == '0':
        dest_path = os.path.join(no_glasses_dir, filename)
    else:
        dest_path = os.path.join(glasses_dir, filename)

    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)

# Verify the split
print("Images with glasses:", len(os.listdir(glasses_dir)))
print("Images without glasses:", len(os.listdir(no_glasses_dir)))
