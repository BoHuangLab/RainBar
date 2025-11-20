import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for interactive plots
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
import datetime


#%% DATA IMPORT AND PRE-PROCESSING

# Load the main data
rb64_subfilter = pd.read_csv('', low_memory=False)

# Set experiment prefix for saved files
experiment_name = ''

# Clean column names
rb64_subfilter.columns = rb64_subfilter.columns.str.lower()

# Separate well variable
rb64_subfilter[['well', 'image']] = rb64_subfilter['image_number'].str.split('-', expand=True)

# Group by and tally
rb64_subfilter.groupby('well').size().reset_index(name='counts')

# Load Rainbar well IDs
rb64_subfilter_ids = pd.read_csv('')

# Merge dataframes
rb64_subfilter = rb64_subfilter.merge(rb64_subfilter_ids, on='well', how='left')
rb64_subfilter['rb_id'] = rb64_subfilter['nls'] + '-' + rb64_subfilter['nes']

# Group by and tally
rb64_subfilter.groupby('rb_id').size().reset_index(name='counts')

# Create compressed key variable
rb64_subfilter['well_image_cell_nucleus'] = rb64_subfilter['image_number'].astype(str) + '-' + rb64_subfilter['cell_number'].astype(str) + '-' + rb64_subfilter['nuclei_number'].astype(str)
rb64_subfilter.drop(['image_number', 'cell_number', 'nuclei_number', 'well', 'image'], axis=1, inplace=True)
rb64_subfilter.insert(0, 'well_image_cell_nucleus', rb64_subfilter.pop('well_image_cell_nucleus'))

# Set a QC filter for low quality cells
qc_rb64_subfilter = rb64_subfilter[(rb64_subfilter['circularity'] < 0.8)]
 # Temporary EGFP-EGFP filter
# Filter the dataset for entries containing "D04" - EGFP-EGFP singles well
d04_filtered = qc_rb64_subfilter[(qc_rb64_subfilter['well_image_cell_nucleus'].str.contains('D04')) & 
                                 (qc_rb64_subfilter['cytoplasm_intensity_ch09'] <= qc_rb64_subfilter['nucleus_intensity_ch09'])]

# Filter out the "D04" entries from the original dataset
qc_rb64_subfilter = qc_rb64_subfilter[~qc_rb64_subfilter['well_image_cell_nucleus'].str.contains('D04')]

# Combine the filtered "D09" entries with the other entries
qc_rb64_subfilter = pd.concat([d04_filtered, qc_rb64_subfilter])
del d04_filtered

# Define the channels to drop
ch_to_drop = ['ch12', 'ch14', 'ch02', 'ch03', 'ch04']
# Create the list of columns to drop using list comprehension
columns_to_drop = [f'cytoplasm_intensity_{ch}' for ch in ch_to_drop] + [f'nucleus_intensity_{ch}' for ch in ch_to_drop]
qc_rb64_subfilter.drop(columns_to_drop, axis=1, inplace=True)

qc_rb64_subfilter = qc_rb64_subfilter[~qc_rb64_subfilter['well_image_cell_nucleus'].str.contains('D04|B02|F06')]

# Save the data
qc_rb64_subfilter.to_parquet(f'out/{experiment_name}_data.parquet')

# Clean up
#del rb64_subfilter_ids, rb64_subfilter, qc_rb64_subfilter


#%% RATIO DATAFRAME GENERATION

# Load the data subset
rb64_subfilter = pd.read_parquet(f'out/{experiment_name}_data.parquet')

# Function to get last 4 characters of a string
def last_4_chars(x):
    return x[-4:]

# Create cytoplasm ratios
cell_ratio_df = rb64_subfilter.dropna().filter(regex="cytoplasm_intensity|nes|well_image_cell_nucleus")

cell_ratio_ids = cell_ratio_df.dropna().filter(['nes', 'well_image_cell_nucleus'])
cell_ratio_df = cell_ratio_df.dropna().drop(columns=['nes', 'well_image_cell_nucleus'])

cell_ratio_list = {}
cell_ratio_names = []

for i in range(cell_ratio_df.shape[1]):
    for j in range(cell_ratio_df.shape[1]):
        if i != j:
            cell_ratio = cell_ratio_df.iloc[:, i] / cell_ratio_df.iloc[:, j]
            col_i_name = last_4_chars(cell_ratio_df.columns[i])
            col_j_name = last_4_chars(cell_ratio_df.columns[j])
            cell_ratio_name = f"{col_i_name}_by_{col_j_name}"
            cell_ratio_list[cell_ratio_name] = cell_ratio
            cell_ratio_names.append(cell_ratio_name)

cell_ratio_df = pd.DataFrame(cell_ratio_list)
cell_ratio_df.columns = ["cell_" + col for col in cell_ratio_df.columns]
cell_ratio_df = cell_ratio_df.assign(nes=cell_ratio_ids['nes'].astype('category'), well_image_cell_nucleus=cell_ratio_ids['well_image_cell_nucleus'])

del cell_ratio_ids, cell_ratio_list, cell_ratio_names

# Create nuclei ratios
nuc_ratio_df = rb64_subfilter.dropna().filter(regex="nucleus_intensity|nls|well_image_cell_nucleus")

nuc_ratio_ids = nuc_ratio_df.dropna().filter(['nls', 'well_image_cell_nucleus'])
nuc_ratio_df = nuc_ratio_df.dropna().drop(columns=['nls', 'well_image_cell_nucleus'])

nuc_ratio_list = {}
nuc_ratio_names = []

for i in range(nuc_ratio_df.shape[1]):
    for j in range(nuc_ratio_df.shape[1]):
        if i != j:
            nuc_ratio = nuc_ratio_df.iloc[:, i] / nuc_ratio_df.iloc[:, j]
            col_i_name = last_4_chars(nuc_ratio_df.columns[i])
            col_j_name = last_4_chars(nuc_ratio_df.columns[j])
            nuc_ratio_name = f"{col_i_name}_by_{col_j_name}"
            nuc_ratio_list[nuc_ratio_name] = nuc_ratio
            nuc_ratio_names.append(nuc_ratio_name)

nuc_ratio_df = pd.DataFrame(nuc_ratio_list)
nuc_ratio_df.columns = ["nuc_" + col for col in nuc_ratio_df.columns]
nuc_ratio_df = nuc_ratio_df.assign(nls=nuc_ratio_ids['nls'].astype('category'), well_image_cell_nucleus=nuc_ratio_ids['well_image_cell_nucleus'])

del nuc_ratio, nuc_ratio_ids, nuc_ratio_list, nuc_ratio_name, nuc_ratio_names, i, j, col_i_name, col_j_name

# Merge cytoplasm and nuclei ratios
rb64sf_ratios = pd.merge(cell_ratio_df, nuc_ratio_df, on="well_image_cell_nucleus").dropna()
rb64sf_ratios['rb_id'] = rb64sf_ratios['nls'].astype(str) + '-' + rb64sf_ratios['nes'].astype(str)
rb64sf_ratios = rb64sf_ratios.drop(columns=['nls', 'nes'])

# Export ratio_df
rb64sf_ratios.to_parquet(f'out/{experiment_name}_ratios.parquet')

# Clean up
del cell_ratio_df, nuc_ratio_df


#%% TRAIN MODEL ON SINGLE RB WELLS

# Read singles wells data
rb64sf_ratios = pd.read_parquet(f'out/{experiment_name}_ratios.parquet')

# Shuffle the DataFrame
shuffled_rb64sf_ratios = rb64sf_ratios.sample(frac=1, random_state=123).reset_index(drop=True)
#shuffled_rb64sf_ratios.to_csv(f'out/{experiment_name}_shuffled_ratios.csv', index=False)

# Print column names to verify
print("RB DataFrame columns:", shuffled_rb64sf_ratios.columns)

# Separate the identifier column
rb_identifiers = shuffled_rb64sf_ratios['well_image_cell_nucleus']

# Drop the identifier column from the dataframe
shuffled_rb64sf_ratios = shuffled_rb64sf_ratios.drop(columns=['well_image_cell_nucleus'])

# Print head of dataframe
print(shuffled_rb64sf_ratios.head())

# Prepare data
def prepare_data(df, target):
    # Drop rows with missing values
    df = df.dropna()
    X = df.drop(target, axis=1).values
    y = df[target].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y, le

# Prepare RB data
X_rb, y_rb, le_rb = prepare_data(shuffled_rb64sf_ratios, 'rb_id')

# Perform train-test split on RB data
X_train_indices, X_test_indices, y_train_indices, y_test_indices = train_test_split(
    np.arange(X_rb.shape[0]), np.arange(X_rb.shape[0]), test_size=0.3, random_state=123, stratify=y_rb
)

# Use the same indices for RB data
X_rb_train, X_rb_test = X_rb[X_train_indices], X_rb[X_test_indices]
y_rb_train, y_rb_test = y_rb[X_train_indices], y_rb[X_test_indices]
rb_train_ids, rb_test_ids = rb_identifiers.iloc[X_train_indices], rb_identifiers.iloc[X_test_indices]

# Standardize the features
scaler = StandardScaler()
X_rb_train = scaler.fit_transform(X_rb_train)
X_rb_test = scaler.transform(X_rb_test)

# Convert integer labels to one-hot encoding
y_rb_train = to_categorical(y_rb_train, num_classes=len(np.unique(y_rb)))
y_rb_test = to_categorical(y_rb_test, num_classes=len(np.unique(y_rb)))

# Multi-Layer Perceptron (MLP)
mlp_model = Sequential([
    Input(shape=(X_rb_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_rb)), activation='softmax')
])

mlp_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train MLP for RB
mlp_model.fit(X_rb_train, y_rb_train, epochs=10, batch_size=32, validation_data=(X_rb_test, y_rb_test))
y_rb_mlp_pred = mlp_model.predict(X_rb_test)
y_rb_mlp_pred_classes = np.argmax(y_rb_mlp_pred, axis=1)
y_rb_test_classes = np.argmax(y_rb_test, axis=1)

# Map the predicted classes back to the original labels for RB
y_rb_test_labels = le_rb.inverse_transform(y_rb_test_classes)
y_rb_mlp_pred_labels = le_rb.inverse_transform(y_rb_mlp_pred_classes)

print(confusion_matrix(y_rb_test_labels, y_rb_mlp_pred_labels))
print(classification_report(y_rb_test_labels, y_rb_mlp_pred_labels))

# Save the model
mlp_model.save(f'out/{experiment_name}_mlp_model.keras')

# Merge predictions back together using 'well_image_cell_nucleus' as the identifier
rb_predictions_df = pd.DataFrame({
    'well_image_cell_nucleus': rb_test_ids.values,
    'True_RB_ID': y_rb_test_labels,
    'MLP_RB_Prediction': y_rb_mlp_pred_labels
})

# Save the DataFrame to a CSV file
rb_predictions_df.to_csv(f'out/{experiment_name}_test_predictions_mlp.csv', index=False)

# Calculate correctness of each prediction
rb_predictions_df['mlp_correct'] = rb_predictions_df['True_RB_ID'] == rb_predictions_df['MLP_RB_Prediction']

# Calculate mean accuracy for MLP predictor
print("MLP Accuracy: ", rb_predictions_df['mlp_correct'].mean())

# Group by reference and calculate accuracy for each group
grouped_accuracy = rb_predictions_df.groupby('True_RB_ID').agg({
    'mlp_correct': 'mean'
}).reset_index()

grouped_accuracy.columns = ['reference', 'accuracy_mlp']
print(grouped_accuracy)

# Save the grouped prediction accuracies
model_metrics = classification_report(y_rb_test_labels, y_rb_mlp_pred_labels, output_dict=True)
model_metrics_df = pd.DataFrame(model_metrics).transpose()
model_metrics_df.to_csv(f'temp2/{experiment_name}_test_prediction_accuracy_mlp.csv', index=True)

# Generate the confusion matrix
cm = confusion_matrix(y_rb_test_labels, y_rb_mlp_pred_labels, labels=le_rb.classes_)

# Convert confusion matrix to DataFrame for better readability
cm_df = pd.DataFrame(cm, index=[f"Reference_{label}" for label in le_rb.classes_],
                     columns=[f"Prediction_{label}" for label in le_rb.classes_])

# Print or save the confusion matrix DataFrame
print(cm_df)

# Save the confusion matrix to a CSV file
cm_df.to_csv(f'out/{experiment_name}_confusion_matrix_mlp.csv')


#%% POOLS DATA IMPORT AND PRE-PROCESSING

# Name the pools
pools_name = ''

# Load the pools data
rb64_subfilter_pools = pd.read_csv('', low_memory=False)

# Clean column names and convert specific columns to numeric
rb64_subfilter_pools.columns = rb64_subfilter_pools.columns.str.lower()

# Separate well variable
rb64_subfilter_pools[['well', 'image']] = rb64_subfilter_pools['image_number'].str.split('-', expand=True)

# Group by and tally
print(rb64_subfilter_pools.groupby('well').size().reset_index(name='counts'))

# Outcome placeholder variable
rb64_subfilter_pools['rb_id'] = np.nan

# Create compressed key variable
rb64_subfilter_pools['well_image_cell_nucleus'] = rb64_subfilter_pools['image_number'].astype(str) + '-' + rb64_subfilter_pools['cell_number'].astype(str) + '-' + rb64_subfilter_pools['nuclei_number'].astype(str)
rb64_subfilter_pools.drop(['image_number', 'cell_number', 'nuclei_number', 'well', 'image'], axis=1, inplace=True)
rb64_subfilter_pools.insert(0, 'well_image_cell_nucleus', rb64_subfilter_pools.pop('well_image_cell_nucleus'))

# Set a QC filter for low quality cells
qc_rb64_subfilter_pools = rb64_subfilter_pools[(rb64_subfilter_pools['circularity'] < 0.8)]

# Save the dataframes to .parquet
rb64_subfilter_pools.to_parquet(f'out/{experiment_name}_{pools_name}_data.parquet')

# GENERATE RATIO DFS
# Function to get last 16 characters of a string
def last_4_chars(x):
    return x[-4:]

# Create cytoplasm ratios
pools_cell_ratio_df = rb64_subfilter_pools.filter(regex="cytoplasm_intensity|well_image_cell_nucleus")

pools_cell_ratio_ids = pools_cell_ratio_df.dropna().filter(['well_image_cell_nucleus'])
pools_cell_ratio_df = pools_cell_ratio_df.dropna().drop(columns=['well_image_cell_nucleus'])

pools_cell_ratio_list = {}
pools_cell_ratio_names = []

for i in range(pools_cell_ratio_df.shape[1]):
    for j in range(pools_cell_ratio_df.shape[1]):
        if i != j:
            pools_cell_ratio = pools_cell_ratio_df.iloc[:, i] / pools_cell_ratio_df.iloc[:, j]
            pools_col_i_name = last_4_chars(pools_cell_ratio_df.columns[i])
            pools_col_j_name = last_4_chars(pools_cell_ratio_df.columns[j])
            pools_cell_ratio_name = f"{pools_col_i_name}_by_{pools_col_j_name}"
            pools_cell_ratio_list[pools_cell_ratio_name] = pools_cell_ratio
            pools_cell_ratio_names.append(pools_cell_ratio_name)

pools_cell_ratio_df = pd.DataFrame(pools_cell_ratio_list)
pools_cell_ratio_df.columns = ["cyto_" + col for col in pools_cell_ratio_df.columns]
pools_cell_ratio_df = pools_cell_ratio_df.assign(well_image_cell_nucleus=pools_cell_ratio_ids['well_image_cell_nucleus'])

# Create nuclei ratios
pools_nuc_ratio_df = rb64_subfilter_pools.filter(regex="nucleus_intensity|well_image_cell_nucleus")

pools_nuc_ratio_ids = pools_nuc_ratio_df.dropna().filter(['well_image_cell_nucleus'])
pools_nuc_ratio_df = pools_nuc_ratio_df.dropna().drop(columns=['well_image_cell_nucleus'])

pools_nuc_ratio_list = {}
pools_nuc_ratio_names = []

for i in range(pools_nuc_ratio_df.shape[1]):
    for j in range(pools_nuc_ratio_df.shape[1]):
        if i != j:
            pools_nuc_ratio = pools_nuc_ratio_df.iloc[:, i] / pools_nuc_ratio_df.iloc[:, j]
            pools_col_i_name = last_4_chars(pools_nuc_ratio_df.columns[i])
            pools_col_j_name = last_4_chars(pools_nuc_ratio_df.columns[j])
            pools_nuc_ratio_name = f"{pools_col_i_name}_by_{pools_col_j_name}"
            pools_nuc_ratio_list[pools_nuc_ratio_name] = pools_nuc_ratio
            pools_nuc_ratio_names.append(pools_nuc_ratio_name)

pools_nuc_ratio_df = pd.DataFrame(pools_nuc_ratio_list)
pools_nuc_ratio_df.columns = ["nuc_" + col for col in pools_nuc_ratio_df.columns]
pools_nuc_ratio_df = pools_nuc_ratio_df.assign(well_image_cell_nucleus=pools_nuc_ratio_ids['well_image_cell_nucleus'])

# Merge cytoplasm and nuclei ratios
rb64sf_pools_ratio = pd.merge(pools_cell_ratio_df, pools_nuc_ratio_df, on="well_image_cell_nucleus").dropna()
rb64sf_pools_ratio['rb_id'] = np.nan

# Export ratio_df
rb64sf_pools_ratio.to_parquet(f'out/{experiment_name}_{pools_name}_ratios.parquet')


#%% PREDICT POOLED WELL RB IDS

# Function to make predictions on new data
def predict_new_data(model_path, new_data_path, scaler, le_rb):
    # Load the saved model
    model = load_model(model_path)
    print("Model loaded from", model_path)
    
    # Read new data
    new_rb_df = pd.read_parquet(new_data_path)
    
    # Print column names to verify
    print("New RB DataFrame columns:", new_rb_df.columns)
    
    # Separate the identifier column
    new_rb_identifiers = new_rb_df[['well_image_cell_nucleus']]
    
    # Drop the identifier and target columns from the dataframe
    new_rb_df = new_rb_df.drop(columns=['well_image_cell_nucleus', 'rb_id'])
    
    # Standardize the features
    X_new_rb = scaler.transform(new_rb_df.values)
    
    # Reshape data for CNN model
    X_new_rb_cnn = X_new_rb.reshape((X_new_rb.shape[0], X_new_rb.shape[1], 1))
    
    # Predict with the loaded model
    y_new_rb_pred = model.predict(X_new_rb_cnn)
    y_new_rb_pred_classes = np.argmax(y_new_rb_pred, axis=1)
    
    # Extract confidence scores
    new_confidence_scores = np.max(y_new_rb_pred, axis=1)
    
    # Map the predicted classes back to the original labels for RB
    y_new_rb_pred_labels = le_rb.inverse_transform(y_new_rb_pred_classes)
    
    # Create DataFrame for predictions
    new_rb_predictions_df = pd.DataFrame({
    'well_image_cell_nucleus': new_rb_identifiers['well_image_cell_nucleus'].values,
    'Predicted_RB_ID': y_new_rb_pred_labels,
    'Confidence': new_confidence_scores
})
    
    return new_rb_predictions_df

# Predict on new data
new_data_path = f'out/{experiment_name}_{pools_name}_ratios.parquet'
rb_model_path = f'out/{experiment_name}_mlp_model.keras'
new_predictions_df = predict_new_data(rb_model_path, new_data_path, scaler, le_rb)

# Save new predictions to CSV
new_predictions_df.to_csv(f'out/{experiment_name}_{pools_name}_predictions_mlp.csv', index=False)

del new_data_path, pools_nuc_ratio, pools_nuc_ratio_name, rb_model_path


#%% T-SNE OF MLP PENULTIMATE LAYER

# Ensure the model is built and called
mlp_model.predict(X_rb_test)

# Print model summary to find the correct penultimate layer name
mlp_model.summary()

# Define the penultimate layer name based on the model summary
penultimate_layer_name = 'dense_30'

# Define the input shape explicitly
input_shape = (X_rb_test.shape[1],)

# Create feature extractor model manually using the functional API
inputs = Input(shape=input_shape)
x = mlp_model.layers[0](inputs)
for layer in mlp_model.layers[1:4]:  # Adjust the range to include layers up to the penultimate layer
    x = layer(x)
feature_extractor = Model(inputs=inputs, outputs=x)

# Extract features for the test dataset
features_rb_test = feature_extractor.predict(X_rb_test)

def plot_pooled_well(well_number, scaler, feature_extractor, mlp_model, le_rb, color_mapping):
    # Load the pooled data
    new_data_path = f'out/{experiment_name}_{pools_name}_ratios.parquet'
    new_rb_df = pd.read_parquet(new_data_path)
    
    # Filter the pooled data for the specified well
    well_df = new_rb_df[new_rb_df['well_image_cell_nucleus'].str.contains(well_number, case=False, na=False)]
    well_df = well_df.drop(columns=['well_image_cell_nucleus', 'rb_id'])
    
    # Process the pooled data similarly to how you processed the test data
    X_well = scaler.transform(well_df.values)
    
    # Extract features from the pooled data using the same feature extractor
    features_well = feature_extractor.predict(X_well)
    
    # Obtain predictions for the pooled data
    predictions_well = mlp_model.predict(X_well)
    predicted_labels_well = np.argmax(predictions_well, axis=1)
    
    # Decode the predictions to their original string labels
    predicted_labels_well_string = le_rb.inverse_transform(predicted_labels_well)
    
    return features_well, predicted_labels_well_string

# Define color mapping for each label
color_mapping = {
    'Sapphire': 'mediumblue',
    'EGFP': 'forestgreen',
    'EYFP': 'gold',
    'mOrange2': 'darkorange',
    'mCherry': 'crimson',
    'mAmetrine': 'rebeccapurple',
    'mPapaya': 'pink',
    'LSSmOrange': 'black'
}

# Extract features for the pooled data from a specific well
well_number = 'H09'
features_well, predicted_labels_well_string = plot_pooled_well(well_number, scaler, feature_extractor, mlp_model, le_rb, color_mapping)

print("Start: ", datetime.datetime.now())

# Apply t-SNE to the combined feature set
features_combined = np.concatenate([features_rb_test, features_well])
tsne_combined = TSNE(n_components=2, perplexity=80, early_exaggeration=20.0, random_state=42, n_jobs=-1)
features_2d_tsne_combined = tsne_combined.fit_transform(features_combined)

# Split the combined t-SNE features back into test and pooled data
features_2d_tsne_test = features_2d_tsne_combined[:features_rb_test.shape[0]]
features_2d_tsne_well = features_2d_tsne_combined[features_rb_test.shape[0]:]

# Convert one-hot encoded labels back to original labels
y_rb_test_labels = np.argmax(y_rb_test, axis=1)
y_rb_test_string_labels = le_rb.inverse_transform(y_rb_test_labels)

print("End t-SNE, Start Test Plot: ", datetime.datetime.now())

# Plot 1: Color-coded test points as circles with an alpha of 1.0
plt.figure(figsize=(15, 10))
ax1 = plt.gca()

# Set the background color of the entire plot
ax1.set_facecolor('white')

# Plot each point with full opacity (alpha=1.0)
for i, label in enumerate(y_rb_test_string_labels):
    label_parts = label.split('-')
    center_color = color_mapping[label_parts[0]]  # Inside color
    outline_color = color_mapping[label_parts[1]]  # Outside/Outline color
    ax1.scatter(features_2d_tsne_test[i, 0], features_2d_tsne_test[i, 1], edgecolors=outline_color, facecolors=center_color, s=100, linewidth=2, alpha=1.0)

plt.title('t-SNE Visualization of MLP Features (Test Data)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

# Create a custom legend for the test data
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[class_name], markeredgecolor=color_mapping[class_name], markersize=10, label=class_name) for class_name in color_mapping.keys()]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
plt.savefig(f'out/{experiment_name}_{well_number}_test_tsne.png', bbox_inches='tight')

print("End Test Plot, Start Pool Plot: ", datetime.datetime.now())

# Plot 2: Test points in gray color with pooled predictions color-coded by their predicted outcome
plt.figure(figsize=(15, 10))
ax2 = plt.gca()

# Set the background color of the entire plot
ax2.set_facecolor('white')

# Plot the test points in gray
for i in range(features_2d_tsne_test.shape[0]):
    ax2.scatter(features_2d_tsne_test[i, 0], features_2d_tsne_test[i, 1], edgecolors='gray', facecolors='gray', s=100, linewidth=2, alpha=1.0)

# Overlay pooled data from the specified well with color-coded triangle markers based on predictions
for i, label in enumerate(predicted_labels_well_string):
    label_parts = label.split('-')
    center_color = color_mapping[label_parts[0]]  # Inside color
    outline_color = color_mapping[label_parts[1]]  # Outside/Outline color
    ax2.scatter(features_2d_tsne_well[i, 0], features_2d_tsne_well[i, 1], edgecolors=outline_color, facecolors=center_color, alpha=0.6, label=f'Pooled Data Well {well_number}', marker='^', s=75, linewidth=2)

plt.title(f't-SNE Visualization of MLP Features (Gray Test Data with Pooled Data from Well {well_number})')
plt.xlabel('Component 1')
plt.ylabel('Component 2')

# Create a custom legend for the pooled data
handles = [plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markeredgecolor='black', markersize=10, label=f'Pooled Data Well {well_number}')]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
plt.savefig(f'out/{experiment_name}_{well_number}_pool_tsne.png', bbox_inches='tight')

print("End: ", datetime.datetime.now())

