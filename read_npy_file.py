import numpy as np

def print_shapes(data, prefix=''):
    print("Here")
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}{key}:")
            print_shapes(value, prefix + '  ')
    elif isinstance(data, list):
        print(f"{prefix}List of length {len(data)}")
        # Check if elements are lists (likely for your image paths and keypoints)
        if len(data) > 0:
            if isinstance(data[0], list):
                print(f"{prefix}  Element type: List, Length of first element: {len(data[0])}")
                if len(data[0]) > 0 and isinstance(data[0][0], list):
                    # Assuming lists of lists are keypoints which are themselves lists of [x, y, confidence]
                    print(f"{prefix}    Keypoints sample shape: [{len(data[0][0])}] elements per keypoint")
            elif isinstance(data[0], str):
                print(f"{prefix}  Element type: str (Assuming paths)")
    elif isinstance(data, np.ndarray):
        print(f"{prefix}Array shape: {data.shape}")
    else:
        print(f"{prefix}Type: {type(data)}, Value: {data}")



def print_ims_details(data):
    ims_data = data.get('ims', [])
    print(f"'ims' contains {len(ims_data)} items.")

    if len(ims_data) > 0:
        first_item = ims_data[0]
        if isinstance(first_item, dict):
            print("Structure of the first item (dictionary keys):")
            for key, value in first_item.items():
                print(f"  {key}: Type={type(value).__name__}, Length={len(value) if hasattr(value, '__len__') else 'Not applicable'}")
        elif isinstance(first_item, list):
            print(f"First item is a list with {len(first_item)} elements.")
            if all(isinstance(x, str) for x in first_item):
                print("  This list contains image paths.")
            elif all(isinstance(x, list) for x in first_item):
                print("  This list contains sublists (likely keypoints).")
                if len(first_item) > 0 and isinstance(first_item[0], list):
                    print(f"    First sublist length (keypoint structure): {len(first_item[0])}")
        else:
            print("First item type:", type(first_item).__name__)

data = np.load('data/dataset/zju_mocap/CoreView_387/annots.npy', allow_pickle=True)
annots = data.item()
# Now you can use 'data' as a normal NumPy array
# print(f"DATA: {data}")
# print_shapes(annots)
# print_ims_details(annots)

param = np.load('data/dataset/zju_mocap/CoreView_387/new_params/0.npy', allow_pickle=True)
print(param.item())