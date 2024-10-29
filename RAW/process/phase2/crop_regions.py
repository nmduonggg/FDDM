import os
import json
import argparse
import numpy as np
from tqdm import tqdm

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2

def write2file(metadata, metadata_file, mode='a'):
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
    with open(metadata_file, mode) as f:
        json.dump(metadata, f, indent=4, cls=NpEncoder)
        
def laplacian_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplac = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    mask_img = cv2.convertScaleAbs(laplac)
    return mask_img

def limit_coordinates(binary_img):
    return np.column_stack(np.where(binary_img > 5))

def generate_random_array(n, sz=5):
    arr = np.zeros(sz, dtype=int)  # Initialize an array of zeros with size `sz`
    remaining_sum = n
    ratios = np.array(range(sz))
    ratios = ratios / ratios.sum()
    for i in range(1, sz):
        arr[i] = int(remaining_sum * ratios[sz - i])
        remaining_sum -= arr[i]
    # arr[1] = remaining_sum
    print(arr)
    return arr


def is_valid_image(image):
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(grayscale, 240, 255, cv2.THRESH_BINARY)
    
    total_pixels = image.shape[0] * image.shape[1]
    white_pixels = np.sum(thresholded == 255)
    white_percentage = (white_pixels / total_pixels) * 100
    
    # Check if the percentage of white pixels is less than 50%
    white_condition = white_percentage < 50
    
    blue_mean = np.array([55, 55, 255])
    blue_mask = np.all(np.abs(image - blue_mean) <= 50, axis=-1)
    blue_percentage = blue_mask.sum() / total_pixels * 100
    blue_condition = blue_percentage < 1
    
    return (blue_condition and white_condition)

def get_random_crop_area(image, crop_size, options=None):
    """Get random coordinates for cropping an image of given crop size."""
    height, width = image.shape[:2]
    
    # Ensure the crop size does not exceed the image dimensions
    crop_width = min(crop_size, width)
    crop_height = min(crop_size, height)
    
    if options is None:
        # Randomly select the top-left corner of the crop
        x = np.random.randint(0, width - crop_width + 1)
        y = np.random.randint(0, height - crop_height + 1)
    
    # Randomly select the top-left corner of the crop in the limited areas
    elif options is not None:
        idx = np.random.randint(0, options.shape[0])
        x = options[idx, 0]
        y = options[idx, 1]
    
    return (x, y, crop_width, crop_height)


def random_proportional_crops(image_path, labels_path, image_filename, label_filename, output_dir, num_crops=10):
    """
    Perform random crops on the image and labels, resize labels to match the image size,
    and save metadata for each crop.
    """
    # Resize the labels image to match the size of the input image
    global crop_index
    start_index = crop_index + 1
    # arr = generate_random_array(num_crops)
    arr = [0, 0, 0, 0, sz]
    metadata = []
    crops = []
    image_filename_no_ext = os.path.splitext(image_filename)[0]
    label_filename_no_ext = os.path.splitext(label_filename)[0]
    
    # Crate separated folders
    image_outdir = os.path.join(output_dir, "images")
    label_outdir = os.path.join(output_dir, "labels")
    os.makedirs(image_outdir, exist_ok=True)
    os.makedirs(label_outdir, exist_ok=True)
    
    
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]
    options = limit_coordinates(laplacian_score(image))
    print(options.shape)
    
    for i, n in enumerate(arr):
        if n==0: continue
        crop_size = 256 * (2 ** i)
        print("Crop size: ", crop_size)
        valid_crop_count = 0   
        condition = (options[:, 0] < original_width - crop_size + 1) & (options[:, 1] < original_height - crop_size + 1)
        options = options[condition]
        
        indices = np.random.choice(range(options.shape[0]), n, replace=False)
        
        for idx in tqdm(indices, total=indices.shape[0]):
        
            x = options[idx, 0]
            y = options[idx, 1]

            cropped_image = image[y:y + crop_size, x:x + crop_size]
            crop_area = (x, y, crop_size, crop_size)
            if is_valid_image(cropped_image):
                crops.append(crop_area)
                valid_crop_count += 1
                crop_index += 1
                cv2.imwrite(os.path.join(image_outdir, image_filename_no_ext + f"-crop-{crop_index}.png"), cropped_image)
    del image
    
    print("Cropping label...")

    labels = cv2.imread(labels_path)
    labels_resized = cv2.resize(labels, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    for id, crop in enumerate(crops):
        x, y, crop_width, crop_height = crop
        cropped_labels = labels_resized[y:y + crop_height, x:x + crop_width]
        cv2.imwrite(os.path.join(label_outdir, label_filename_no_ext + f"-crop-{start_index + id}.png"), cropped_labels)
        metadata.append({
            'crop_index': start_index + id,
            'image_filename': image_filename_no_ext,
            'label_filename': label_filename_no_ext,
            'croped_image_filename': image_filename_no_ext + f"-crop-{start_index + id}.png",
            'croped_label_filename': label_filename_no_ext + f"-crop-{start_index + id}.png",
            'crop_coordinates': {
                'x': x,
                'y': y,
                'width': crop_width,
                'height': crop_height
            }
        })
    del labels
    return metadata
    # Generate the random array to define the crop sizes
    

if __name__ == "__main__":

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Random proportional image cropping tool.")

    # Add arguments
    parser.add_argument('--image_folder', default='./images', type=str, help='Path to the folder containing images.')
    parser.add_argument('--labels_folder', default='./labels', type=str, help='Path to the folder containing labels.')
    parser.add_argument('--n', default=6, type=int, help='Number of crops to generate.')
    parser.add_argument('--output_folder', default='./output', type=str, help='Path to the output folder for cropped images and metadata.')
    parser.add_argument('--start_metadata', type=str, help='Previous metadata to get the last index')

    # Parse the arguments
    args = parser.parse_args()

    # Display the parsed arguments
    print(f"Image folder: {args.image_folder}")
    print(f"Labels folder: {args.labels_folder}")
    print(f"Number of crops (n): {args.n}")
    print(f"Output folder: {args.output_folder}")

    os.makedirs(args.output_folder, exist_ok=True)
    
    skip_cases = ["Case_6", "Case_8", "Case_6_1"]
    done_cases = []
    chosen_cases = [f"Case_{i}" for i in [1, 2, 3, 4, 5, 7, 10]]
    # chosen_cases = ["Case_9"]
    print(f"Skip cases: {skip_cases}")
    
    
    crop_index = 0
    if args.start_metadata is not None:
        with open(args.start_metadata, 'r') as f:
            init_meta = json.load(f)
        crop_index = init_meta[-1]['crop_index'] + 1
        
    # crop_index = 79571
        
    print("Crop index: ", crop_index)

    for case_folder in os.listdir(args.labels_folder):
        if (case_folder in skip_cases) or (case_folder in done_cases) or (case_folder not in chosen_cases):
            continue
        labels_folder = os.path.join(args.labels_folder, case_folder)
        images_folder = os.path.join(args.image_folder, case_folder)
        output_folder = os.path.join(args.output_folder, case_folder)
        os.makedirs(output_folder, exist_ok=True)
        
        metadata_file = os.path.join(output_folder, 'metadata.json')

        if not os.path.isdir(labels_folder) or not os.path.isdir(images_folder):
            continue
        
        print(f"Processing images in folder: {case_folder}")
        metadata = []       
        write2file(metadata, metadata_file, 'w')
        
        for label_name in os.listdir(labels_folder):
            if 'slide-' not in label_name: continue
            print("Processing: ", label_name)
            
            # skip indices of mask -> store to process in image
            skip_indices = []
            upsample=False
            
            if "x8" in label_name:
                img_name = label_name.split("-x8")[0] + '.png'
                upsample = True
            else:
                img_name = label_name.split("-labels")[0] + '.png'

            image_path = os.path.join(images_folder, img_name)
            label_path = os.path.join(labels_folder, label_name)
            result_metadata = random_proportional_crops(image_path, label_path, img_name, label_name, output_folder, args.n)
            metadata.extend(result_metadata)
            write2file(metadata, metadata_file, 'w')
        
        # Save the metadata to a JSON file
        write2file(metadata, metadata_file, 'w')


    # # Save the metadata to a JSON file
    # metadata_file = os.path.join(args.output_folder, 'metadata.json')
    # with open(metadata_file, 'w') as f:
    #     json.dump(metadata, f, indent=4)

