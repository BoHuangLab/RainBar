from cellpose import models, io
import numpy as np
import os
import re
from skimage.measure import regionprops, label
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, center_of_mass
import pandas as pd
from matplotlib import colors
from skimage.segmentation import find_boundaries
import gc
from multiprocessing import Pool, cpu_count

# Define the path to your multi-channel images and experiment name
image_dir = '' 
experiment_name = ''

# Define the rows and columns for a 96-well plate
rows = [f'r{str(i).zfill(2)}' for i in range(1, 9)]
cols = [f'c{str(i).zfill(2)}' for i in range(1, 13)]

# Get all unique fields of view (f01, f02, etc.) in the directory
field_numbers = sorted(set(f.split('f')[1][:2] for f in os.listdir(image_dir) if f.endswith('.tiff')))

# Paths to custom Cellpose models
custom_model_path_1 = 'models/custom_cellpose_model_path_1' # Cell body custom segmentation model
custom_model_path_2 = 'models/custom_cellpose_model_path_2' # Nucleus custom segmentation model

# Initialize an empty list to collect data
all_data = []

# Regex pattern to match filenames containing 'ch' followed by digits
pattern = re.compile(r'(.*ch)([0-9]+)(.*)')

# Iterate over all files in the directory
for filename in os.listdir(image_dir):
    match = pattern.match(filename)
    if match:
        prefix = match.group(1)  # Part before the number
        number = match.group(2)  # The numeric part after 'ch'
        suffix = match.group(3)  # Part after the number

        # Format number with two digits
        new_number = f"{int(number):02d}"

        # Construct new filename
        new_filename = f"{prefix}{new_number}{suffix}"

        # Full paths for renaming
        old_file = os.path.join(image_dir, filename)
        new_file = os.path.join(image_dir, new_filename)

        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed {filename} to {new_filename}")

# Helper function to convert row and column to well identifier
def well_identifier(row, col):
    row_letter = chr(ord('A') + int(row[1:]) - 1)
    col_number = int(col[1:])
    return f'{row_letter}{str(col_number).zfill(2)}'

# Function to remove masks touching the edge
def remove_edge_touching_objects(mask):
    mask_copy = mask.copy()
    edge_mask = np.zeros_like(mask, dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_touching_labels = np.unique(mask_copy[edge_mask])
    for label in edge_touching_labels:
        mask_copy[mask_copy == label] = 0
    return mask_copy

def process_well_field(args):
    row, col, field_number, image_dir, custom_model_path_1, custom_model_path_2 = args
    well_id = well_identifier(row, col)
    print(f'Processing well: {well_id}, field of view: {field_number}')

    # Define the filenames for the current well and field of view
    image_filenames = [f'{row}{col}f{field_number}p01-ch{str(i).zfill(2)}sk1fk1fl1.tiff' for i in range(1, 15)]
    
    # Load the images into a list
    images = [io.imread(os.path.join(image_dir, filename)) for filename in image_filenames if os.path.exists(os.path.join(image_dir, filename))]

    if not images:
        print(f'  No images found for well {well_id} and field {field_number}. Skipping.')
        return []

    # Example: Maximum intensity projection
    projection = np.max(images, axis=0)

    # SEGMENTATION
    image_ch01 = images[0]
    model_cell = models.CellposeModel(gpu=False, pretrained_model=custom_model_path_1)
    channels_cell = [0, 0]  # Adjust channels as needed
    results_cell = model_cell.eval(image_ch01, channels=channels_cell, diameter=30, flow_threshold=0.4, cellprob_threshold=0.0, min_size=15)

    if len(results_cell) == 3:
        masks_cell, flows_cell, styles_cell = results_cell
        diams_cell = None
    elif len(results_cell) == 4:
        masks_cell, flows_cell, styles_cell, diams_cell = results_cell
    else:
        raise ValueError("Unexpected number of return values from model.eval()")

    masks_cell = remove_edge_touching_objects(masks_cell)
    output_path_cell = f'out/{experiment_name}_cell_masks_{well_id}_{field_number}.tif'
    io.imsave(output_path_cell, masks_cell.astype(np.uint16))
    print(f'  Segmentation of the original image {well_id} fov{field_number} complete.') #Results saved to:', output_path_cell)

    inverted_image = np.invert(image_ch01)
    model_nucleus = models.CellposeModel(gpu=False, pretrained_model=custom_model_path_2)
    channels_nucleus = [0, 0]
    results_nucleus = model_nucleus.eval(inverted_image, channels=channels_nucleus, diameter=30, flow_threshold=0.4, cellprob_threshold=0.0, min_size=15)

    if len(results_nucleus) == 3:
        masks_nucleus, flows_nucleus, styles_nucleus = results_nucleus
        diams_nucleus = None
    elif len(results_nucleus) == 4:
        masks_nucleus, flows_nucleus, styles_nucleus, diams_nucleus = results_nucleus
    else:
        raise ValueError("Unexpected number of return values from model.eval()")

    masks_nucleus = remove_edge_touching_objects(masks_nucleus)
    output_path_nucleus = f'out/{experiment_name}_nucleus_masks_{well_id}_f{field_number}.tif'
    io.imsave(output_path_nucleus, masks_nucleus.astype(np.uint16))
    print(f'  Segmentation of the inverted image {well_id} fov{field_number} complete.') #Results saved to:', output_path_nucleus)

    # RELATE OBJECTS
    output_path_base = 'out/'
    output_path_filtered_nuclei = f'out/filtered_nuclei_masks_{well_id}_f{field_number}.tif'
    output_path_filtered_cells = f'out/filtered_cell_masks_{well_id}_f{field_number}.tif'
    output_path_cytoplasm = f'out/cytoplasm_masks_{well_id}_f{field_number}.tif'
    output_path_shrink_expand_nuclei = f'out/shrink_expand_nuclei_masks_{well_id}_f{field_number}.tif'

    if masks_nucleus.ndim != 2 or masks_cell.ndim != 2:
        raise ValueError("Masks must be 2D arrays.")

    filtered_nuclei_masks = np.zeros_like(masks_nucleus)
    filtered_cells_masks = np.zeros_like(masks_cell)

    labeled_nuclei = label(masks_nucleus)
    labeled_cells = label(masks_cell)

    nuclei_props = regionprops(labeled_nuclei)
    cells_props = regionprops(labeled_cells)

    cell_to_nuclei = {}

    for nucleus in nuclei_props:
        nucleus_label = nucleus.label
        nucleus_bbox = nucleus.bbox
        nucleus_mask = (labeled_nuclei == nucleus_label)

        paired = False
        for cell in cells_props:
            cell_label = cell.label
            cell_bbox = cell.bbox
            cell_mask = (labeled_cells == cell_label)

            if (nucleus_bbox[0] >= cell_bbox[0] and nucleus_bbox[2] <= cell_bbox[2] and
                nucleus_bbox[1] >= cell_bbox[1] and nucleus_bbox[3] <= cell_bbox[3]):

                overlap = nucleus_mask & cell_mask
                if np.any(overlap):
                    filtered_nuclei_masks[nucleus_mask] = nucleus_label
                    filtered_cells_masks[cell_mask] = cell_label
                    cell_to_nuclei[cell_label] = nucleus_label
                    paired = True
                    break

    for cell in cells_props:
        cell_label = cell.label
        if cell_label not in cell_to_nuclei:
            filtered_cells_masks[filtered_cells_masks == cell_label] = 0

    if not cell_to_nuclei:
        print("  No nucleus-cell pairs were found.")

    cytoplasm_masks = np.zeros_like(filtered_cells_masks)
    structure = generate_binary_structure(2, 1)
    
    for cell_label in np.unique(filtered_cells_masks):
        if cell_label == 0:
            continue
        cell_mask = (filtered_cells_masks == cell_label)
        shrunk_cell_mask_for_cytoplasm = binary_erosion(cell_mask, structure=structure, iterations=2)
        nucleus_label = cell_to_nuclei.get(cell_label)
        if nucleus_label:
            nucleus_mask = (filtered_nuclei_masks == nucleus_label)
            expanded_nucleus_mask_for_cytoplasm = binary_dilation(nucleus_mask, structure=structure, iterations=2)
            cytoplasm_mask = shrunk_cell_mask_for_cytoplasm & ~expanded_nucleus_mask_for_cytoplasm
            cytoplasm_masks[cytoplasm_mask] = cell_label

    # Save the cytoplasm mask image
    #io.imsave(output_path_cytoplasm, cytoplasm_masks.astype(np.uint16))
    #print(f'  {well_id} fov{field_number} cytoplasm masks saved to:', output_path_cytoplasm)

    # Generate size-normalized nuclei masks
    shrink_expand_nuclei_masks = np.zeros_like(filtered_nuclei_masks)
    
    for nucleus_label in np.unique(filtered_nuclei_masks):
        if nucleus_label == 0:
            continue
        nucleus_mask = (filtered_nuclei_masks == nucleus_label)
        centroid = center_of_mass(nucleus_mask)
        centroid = tuple(map(int, centroid))
        reduced_nucleus = np.zeros_like(nucleus_mask)
        reduced_nucleus[centroid] = 1
        expanded_nucleus = binary_dilation(reduced_nucleus, structure=structure, iterations=5)
        shrink_expand_nuclei_masks[expanded_nucleus] = nucleus_label

    print(f'  {well_id} fov{field_number} relating and filtering complete.') #Results saved to:', output_path_base)

    data = []
    for cell_label, nucleus_label in cell_to_nuclei.items():
        cell_mask = (filtered_cells_masks == cell_label)
        expanded_nucleus_mask = (shrink_expand_nuclei_masks == nucleus_label)
        original_nucleus_mask = (filtered_nuclei_masks == nucleus_label)
        cytoplasm_mask = (cytoplasm_masks == cell_label)

        # Calculate areas
        cell_area = np.count_nonzero(cell_mask)
        nucleus_area = np.count_nonzero(original_nucleus_mask)
        shrunken_cell_area = np.count_nonzero(shrunk_cell_mask_for_cytoplasm)
        expanded_nucleus_area = np.count_nonzero(expanded_nucleus_mask_for_cytoplasm)
        cytoplasm_area = np.count_nonzero(cytoplasm_mask)

       # Filter out zero-area cytoplasms
        if cytoplasm_area == 0:
           continue

       # Calculate total intensities
        cytoplasm_intensities = []
        nucleus_intensities = []
        for i, image in enumerate(images):
            cytoplasm_intensity = np.mean(image[cytoplasm_mask])
            nucleus_intensity = np.mean(image[expanded_nucleus_mask])
            cytoplasm_intensities.append(cytoplasm_intensity)
            nucleus_intensities.append(nucleus_intensity)
        
        # Calculate shape metrics for the current cell
        cell_props = next(cell for cell in cells_props if cell.label == cell_label)
        circularity = (4 * np.pi * cell_props.area) / (cell_props.perimeter ** 2)
        aspect_ratio = cell_props.major_axis_length / cell_props.minor_axis_length
        eccentricity = cell_props.eccentricity

        data.append({
            "IMAGE_NUMBER": f"{well_id}-{int(field_number)}",
            "CELL_NUMBER": cell_label,
            "NUCLEI_NUMBER": nucleus_label,
            "CIRCULARITY": circularity,
            "ASPECT_RATIO": aspect_ratio,
            "ECCENTRICITY": eccentricity,
            "CYTOPLASM_AREA": cytoplasm_area,
            "CYTOPLASM_INTENSITY_CH01": cytoplasm_intensities[0],
            "CYTOPLASM_INTENSITY_CH02": cytoplasm_intensities[1],
            "CYTOPLASM_INTENSITY_CH03": cytoplasm_intensities[2],
            "CYTOPLASM_INTENSITY_CH04": cytoplasm_intensities[3],
            "CYTOPLASM_INTENSITY_CH05": cytoplasm_intensities[4],
            "CYTOPLASM_INTENSITY_CH06": cytoplasm_intensities[5],
            "CYTOPLASM_INTENSITY_CH07": cytoplasm_intensities[6],
            "CYTOPLASM_INTENSITY_CH08": cytoplasm_intensities[7],
            "CYTOPLASM_INTENSITY_CH09": cytoplasm_intensities[8],
            "CYTOPLASM_INTENSITY_CH10": cytoplasm_intensities[9],
            "CYTOPLASM_INTENSITY_CH11": cytoplasm_intensities[10],
            "CYTOPLASM_INTENSITY_CH12": cytoplasm_intensities[11],
            "CYTOPLASM_INTENSITY_CH13": cytoplasm_intensities[12],
            "CYTOPLASM_INTENSITY_CH14": cytoplasm_intensities[13],
            "NUCLEUS_INTENSITY_CH01": nucleus_intensities[0],
            "NUCLEUS_INTENSITY_CH02": nucleus_intensities[1],
            "NUCLEUS_INTENSITY_CH03": nucleus_intensities[2],
            "NUCLEUS_INTENSITY_CH04": nucleus_intensities[3],
            "NUCLEUS_INTENSITY_CH05": nucleus_intensities[4],
            "NUCLEUS_INTENSITY_CH06": nucleus_intensities[5],
            "NUCLEUS_INTENSITY_CH07": nucleus_intensities[6],
            "NUCLEUS_INTENSITY_CH08": nucleus_intensities[7],
            "NUCLEUS_INTENSITY_CH09": nucleus_intensities[8],
            "NUCLEUS_INTENSITY_CH10": nucleus_intensities[9],
            "NUCLEUS_INTENSITY_CH11": nucleus_intensities[10],
            "NUCLEUS_INTENSITY_CH12": nucleus_intensities[11],
            "NUCLEUS_INTENSITY_CH13": nucleus_intensities[12],
            "NUCLEUS_INTENSITY_CH14": nucleus_intensities[13]
        })

    # Annotate and save the image
    image = images[0]
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image, cmap='gray')

    cell_boundaries = find_boundaries(filtered_cells_masks, mode='outer')
    nucleus_boundaries = find_boundaries(filtered_nuclei_masks, mode='outer')

    cell_boundary_mask = np.zeros_like(image, dtype=bool)
    nucleus_boundary_mask = np.zeros_like(image, dtype=bool)
    cell_boundary_mask[cell_boundaries] = True
    nucleus_boundary_mask[nucleus_boundaries] = True

    ax.imshow(np.ma.masked_where(~cell_boundary_mask, cell_boundary_mask), cmap=colors.ListedColormap(['yellow']), alpha=0.9)
    ax.imshow(np.ma.masked_where(~nucleus_boundary_mask, nucleus_boundary_mask), cmap=colors.ListedColormap(['cyan']), alpha=0.9)

    for cell in cells_props:
        cell_label = cell.label
        if cell_label in cell_to_nuclei:
            y, x = cell.centroid
            ax.text(x, y, str(cell_label), color='yellow', fontsize=6, ha='center', va='center', weight='bold')

    for nucleus in nuclei_props:
        nucleus_label = nucleus.label
        if nucleus_label in cell_to_nuclei.values():
            y, x = nucleus.centroid
            ax.text(x + 5, y + 5, str(nucleus_label), color='cyan', fontsize=6, ha='center', va='center', weight='bold')

    output_image_path = f'out/{experiment_name}_annotated_image_ch01_{well_id}_f{field_number}.png'
    plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Annotated image {well_id} fov{field_number} saved to {output_image_path}")

    # Clear variables to free up memory
    del images, projection, masks_cell, flows_cell, styles_cell, masks_nucleus, flows_nucleus, styles_nucleus
    del filtered_nuclei_masks, filtered_cells_masks, cytoplasm_masks, shrink_expand_nuclei_masks
    del labeled_nuclei, labeled_cells, nuclei_props, cells_props, cell_to_nuclei
    gc.collect()

    return data

if __name__ == '__main__':
    # Extract the first 6 characters (well identifiers) from filenames and keep unique values
    present_wells = set(f[:6] for f in os.listdir(image_dir) if f.endswith('.tiff'))
    print(f'Present wells: {present_wells}')

    args_list = [(row, col, field_number, image_dir, custom_model_path_1, custom_model_path_2)
                 for row in rows
                 for col in cols
                 for field_number in field_numbers
                 if f'{row}{col}' in present_wells]

    #print(f'Arguments list: {args_list}')

    # Define the number of worker processes
    num_workers = min(6, cpu_count())  # Use up to 6 cores or the number of available CPU cores, whichever is smaller

    # Initialize the Pool and run the worker function in parallel
    with Pool(num_workers) as pool:
        results = pool.map(process_well_field, args_list)

    # Combine all individual results into a single DataFrame
    all_data = [item for sublist in results for item in sublist]

    # Create a DataFrame from the collected data
    if all_data:
        df = pd.DataFrame(all_data)
    else:
        df = pd.DataFrame()
    print(df)
    
    # Filter out rows with cytoplasm area smaller than 10
    df = df[df["CYTOPLASM_AREA"] >= 10]

    # Save the final compiled DataFrame to ensure all data is captured
    output_path_final_dataframe = f'out/{experiment_name}_intensity_measurements.csv'
    df.to_csv(output_path_final_dataframe, index=False)
    print(f"Final DataFrame saved to {output_path_final_dataframe}")
