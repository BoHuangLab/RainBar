import cv2
import numpy as np
from cellpose import models, io
from sklearn.neighbors import NearestNeighbors
import os
import re

def segment_cells(image_path, model):
    image = io.imread(image_path)
    result = model.eval(image, channels=[0, 0], diameter=30, flow_threshold=0.4, cellprob_threshold=0.0, min_size=15)
    if len(result) == 4:
        masks, flows, styles, diams = result
    else:
        masks, flows, styles = result
    return masks

def get_centroids(masks):
    centroids = []
    for mask in np.unique(masks):
        if mask == 0:  # Skip background
            continue
        y, x = np.where(masks == mask)
        centroid = (int(np.mean(x)), int(np.mean(y)))
        centroids.append(centroid)
    return centroids

model = models.CellposeModel(gpu=False, pretrained_model='') # Enter the location of your pre-trained cell sesgmentation model

folder1 = ''  # Time course folder with "sk60" images
folder2 = ''  # Debarcoding folder with "sk1" images
output_folder = ''

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all files in both folders
files_folder1 = sorted(os.listdir(folder1))
files_folder2 = sorted(os.listdir(folder2))

# Define a regex pattern to match filenames containing either "ch01" or "ch1" but not "ch10"
pattern_ch = re.compile(r'ch0?1(?!\d)')

# Filter and match files based on the first nine characters and specific substrings
matched_files = {}
for file1 in files_folder1:
    if "sk60" in file1 and pattern_ch.search(file1):
        key = file1[:9]
        matched_files[key] = {'file1': file1, 'file2': None}

for file2 in files_folder2:
    if "sk1" in file2 and pattern_ch.search(file2):
        key = file2[:9]
        if key in matched_files:
            matched_files[key]['file2'] = file2

# Process each matched pair
for key, files in matched_files.items():
    if files['file1'] and files['file2']:
        image_path1 = os.path.join(folder1, files['file1'])
        image_path2 = os.path.join(folder2, files['file2'])

        # Segment cells in both images using your custom Cellpose model
        masks1 = segment_cells(image_path1, model)
        masks2 = segment_cells(image_path2, model)

        # Extract the centroids
        centroids1 = get_centroids(masks1)
        centroids2 = get_centroids(masks2)

        # Convert centroids to numpy arrays
        points_image1 = np.array(centroids1)
        points_image2 = np.array(centroids2)

        # Use NearestNeighbors to find the closest points
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points_image2)
        distances, indices = nbrs.kneighbors(points_image1)

        # Filter matches by a distance threshold to remove outliers
        distance_threshold = 20  # adjust this value as needed
        good_matches = distances[:, 0] < distance_threshold

        # Extract matched points
        matched_points_image1 = points_image1[good_matches]
        matched_points_image2 = points_image2[indices[good_matches].flatten()]

        # Estimate the affine transformation matrix
        matrix, inliers = cv2.estimateAffinePartial2D(matched_points_image2, matched_points_image1)

        # Apply the transformation to the image
        image2 = io.imread(image_path2)
        transformed_image = cv2.warpAffine(image2, matrix, (image2.shape[1], image2.shape[0]))

        # Save the transformed image
        #transformed_image_path = os.path.join(output_folder, f'transformed_{files["file2"]}')
        #cv2.imwrite(transformed_image_path, transformed_image)

        # Transform the mask
        transformed_mask = cv2.warpAffine(masks2.astype(np.uint8), matrix, (masks2.shape[1], masks2.shape[0]))
        transformed_mask_path = os.path.join(output_folder, f'transformed_mask_{files["file2"]}')
        io.imsave(transformed_mask_path, transformed_mask)

        # Load the reference mask for comparison
        mask1 = masks1.astype(np.uint8)

        # Overlay masks for comparison
        overlay = cv2.addWeighted(mask1, 0.5, transformed_mask, 0.5, 0)

        # Save the overlay image
        overlay_path = os.path.join(output_folder, f'overlay_{files["file2"]}')
        io.imsave(overlay_path, overlay)

print("Processing complete.")
