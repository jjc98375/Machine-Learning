from datasets import load_dataset

# Load the dataset
dataset = load_dataset("Shelton1013/SwitchLingua_text")

# Check the structure
print(dataset)

# Access specific splits (if available)
# train_data = dataset['train']
# test_data = dataset['test']

# View a sample
print(dataset['train'][0])  # Adjust based on actual split names