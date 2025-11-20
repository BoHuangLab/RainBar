# RAINBAR: A Live-Cell CRISPR Screening Platform

## Description
--INSERT PROJECT DESCRIPTION--

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)
- [Contact](#contact)

## Installation

- Install conda
- Create RainBar virtual environment with: conda create --name RainBar, then conda activate RainBar to activate the venv
- pip/conda install the following packages: cellpose numpy scikit-image matplotlib scipy pandas scikit-learn keras tensorflow
- also install or ensure you can import: os, re, gc, multiprocessing, datetime

## Usage

- Open _op_image_analysis.py and set these variables for your image analysis: 
  - image_dir -- point the image analysis script to the directory containing the images you wish to segment and quantify.
  - experiment_name -- use this variable to give a descriptive name to all output files.
  - custom_model_path[s] -- point the image analysis script to your cell and nucleus segmentation models.
  - additionally, you can set which intermediate files (masks, annotations) you would like to save and their output destinations by scrolling through the script and modifying these lines (io.imsave).
  - num_workers -- you may also adjust parallelization using this variable near the bottom of the script.
- Open _op_rb64_analysis_mlp.py and set these variables for your data analysis:
  - Cell 1 (Singles Data Import and Pre-Process):
    - rb64_subfilter (main data) -- point the data analysis script to the intensity measurements you generated in the previous step.
    - experiment_name -- use this variable to give a descriptive name to all output files.
    - rb64_subfilter_ids -- create a .csv file with the identities of your single construct wells and provide this path here.
    - qc_rb64_subfilter -- modify circularity, area, or others filters as necessary for your analysis.
    - ch_to_drop -- choose which channels you want to (optionally) drop from the dataset.
    - optionally save the intermediate output.
  - Cell 2 (Ratio Dataframe Generation):
    - if not saving the intermediate output from the previous step, use the variable name for your main data.
    - optionally save the intermediate output.
  - Cell 3 (Training MLP Model on Singles Wells):
    - if not saving the intermediate output from the previous step, use the variable name for your ratio data.
    - this cell will give you information to evaluate your model on 30% withheld test data.
  - Cell 4 (Pooled Data Import, Pre-Process, and Ratio Dataframe Generation):
    - analyze your pooled wells in the same fashion as your singles using the _op_image_analysis.py script.
    - pools_name -- use this variable to give a descriptive name to all output files.
    - rb64_subfilter_pools -- point the data analysis script to the pools intensity measurements you generated in the previous step.
    - optionally drop low quality wells as well.
    - optionally save the intermediate output.
  - Cell 5 (Pooled Data Predictions):
    - if not saving the intermediate output from the previous step, use the variable name for your pools data.
  - Cell 6 (T-SNE Visualization):
    - well_number -- set the pools well number you wish to visualize over grayed-out test data.
    - produce a visualization from the penultimate layer of the MLP model with the first plot as a T-SNE of the 30% withheld test data predictions, and the second plot as the pooled well of your choice overlaid on grayed-out test data
  
