import pandas as pd
import os

# Load CSV files
val_df = pd.read_csv('validation-images-with-rotation.csv')
test_df = pd.read_csv('test-images-with-rotation.csv')

# Create data directory if it doesn't exist
data_dir = 'data/Open_Images'
os.makedirs(data_dir, exist_ok=True)

out_file = f'{data_dir}/open_images_image_list.txt'
# Generate output file
with open(out_file, 'w') as f:
    # Write validation images
    for image_id in val_df['ImageID']:
        f.write(f'validation/{image_id}\n')

    # Write test images
    for image_id in test_df['ImageID']:
        f.write(f'test/{image_id}\n')

print(f"Generated {out_file} with {len(val_df)} validation and {len(test_df)} test images")
