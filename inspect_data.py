import pickle

# Choose a file to inspect
# Make sure to use the correct path separator for your OS (e.g., '\\' or '/')
file_path = 'd:\\SmartGuard\\data\\an_data\\an_trn_instance_10.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Successfully loaded {file_path}")
    print(f"Data type: {type(data)}")

    if isinstance(data, list):
        print(f"Number of sequences: {len(data)}")
        if len(data) > 0:
            print(f"First sequence type: {type(data[0])}")
            print(f"First sequence: {data[0]}")
            if isinstance(data[0], list) and len(data[0]) > 0:
                print(f"First event in first sequence: {data[0][0]}")
        # You can add more prints here to explore deeper
        # For example, print a few more sequences or events
        # print(f"Second sequence: {data[1]}") 
    elif isinstance(data, dict):
        print(f"Data is a dictionary. Keys: {list(data.keys())}")
        # Explore dictionary content based on its keys
    else:
        print(f"Data content: {data}")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")