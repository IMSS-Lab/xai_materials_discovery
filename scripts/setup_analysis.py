"""
Setup script for GNoME XAI analysis with actual model training (simplified version).

This script prepares the environment by:
1. Downloading the GNoME dataset
2. Training a simple GNoME model for 150 epochs with learning rate reduction
3. Saving the trained model to the models directory

Run this script before running main_simplified.py.
"""

import os
import sys
import numpy as np
import logging
import json
from pathlib import Path
import shutil
import subprocess
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import jraph
import matplotlib.pyplot as plt
import pandas as pd
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = 'data'
MODEL_DIR = 'models'
NUM_ELEMENTS = 94  # Maximum number of elements in the periodic table
TRAINING_EPOCHS = 150  # Number of training epochs


def download_data(data_dir=DATA_DIR):
    """Download GNoME data using the provided scripts."""
    logger.info("Checking if data needs to be downloaded...")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if data has already been downloaded
    gnome_data_dir = os.path.join(data_dir, 'gnome_data')
    if os.path.exists(gnome_data_dir) and os.listdir(gnome_data_dir):
        logger.info("Data already downloaded. Skipping download.")
        return
    
    logger.info("Downloading data...")
    
    # Use subprocess to call the download scripts instead of importing them
    # This avoids the flag conflicts between the two scripts
    
    # Try to use Google Cloud Storage API first
    try:
        logger.info("Attempting to download using Google Cloud Storage API...")
        result = subprocess.run(
            [sys.executable, "scripts/download_data_cloud.py", f"--data_dir={data_dir}"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Download completed successfully using Google Cloud Storage API.")
        logger.info(result.stdout)
        return
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"Could not use Google Cloud Storage API: {str(e)}")
        if hasattr(e, 'stderr'):
            logger.warning(e.stderr)
        logger.info("Falling back to wget method...")
    
    # Fall back to wget method
    try:
        logger.info("Attempting to download using wget...")
        result = subprocess.run(
            [sys.executable, "scripts/download_data_wget.py", f"--data_dir={data_dir}"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Download completed successfully using wget.")
        logger.info(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Failed to download data: {str(e)}")
        if hasattr(e, 'stderr'):
            logger.error(e.stderr)
        
        # Try a direct wget approach as last resort
        try:
            logger.info("Attempting direct download with wget...")
            os.makedirs(os.path.join(data_dir, 'gnome_data'), exist_ok=True)
            
            # Download a few key files directly
            base_url = "https://storage.googleapis.com/gdm_materials_discovery/gnome_data"
            files_to_download = [
                "stable_materials_summary.csv",
            ]
            
            for filename in files_to_download:
                url = f"{base_url}/{filename}"
                output_path = os.path.join(data_dir, 'gnome_data', filename)
                subprocess.run(["wget", url, "-O", output_path], check=True)
            
            logger.info("Basic files downloaded successfully via direct wget.")
        except Exception as direct_e:
            logger.error(f"All download methods failed. Last error: {str(direct_e)}")
            raise


def load_structure_data(data_dir=DATA_DIR, limit=100):
    """Load structure data from GNoME dataset."""
    logger.info("Loading structure data from GNoME dataset...")
    
    # Path to summary file
    summary_file = os.path.join(data_dir, 'gnome_data', 'stable_materials_summary.csv')
    
    # Check if the file exists
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}. Make sure data is downloaded.")
    
    # Load the summary CSV
    logger.info(f"Loading data from {summary_file}")
    df = pd.read_csv(summary_file)
    
    # Limit the number of structures
    df = df.head(limit)
    
    logger.info(f"Loaded {len(df)} structures from summary file.")
    return df


def create_graph_from_structure(structure_data):
    """Convert structure data to a graph representation for GNoME model."""
    # Extract formula
    formula = structure_data.get('Reduced Formula', 'Si')
    
    # Extract element symbols from formula
    element_symbols = re.findall(r'([A-Z][a-z]?)', formula)
    
    # If no elements found, use silicon as default
    if not element_symbols:
        element_symbols = ['Si']
    
    # Use up to 8 atoms for simplicity
    n_atoms = min(8, len(element_symbols) * 2)
    
    # Pad element symbols if needed
    while len(element_symbols) < n_atoms:
        element_symbols.append(element_symbols[0])
    
    # Element to index mapping (simplified)
    element_to_idx = {
        'H': 0, 'He': 1, 'Li': 2, 'Be': 3, 'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9,
        'Na': 10, 'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14, 'S': 15, 'Cl': 16, 'Ar': 17,
        'K': 18, 'Ca': 19, 'Sc': 20, 'Ti': 21, 'V': 22, 'Cr': 23, 'Mn': 24, 'Fe': 25,
        'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29, 'Ga': 30, 'Ge': 31, 'As': 32, 'Se': 33
    }
    
    # Create node features
    nodes = np.zeros((n_atoms, NUM_ELEMENTS))
    for i in range(n_atoms):
        element = element_symbols[i % len(element_symbols)]
        # Default to hydrogen if not in our mapping
        idx = element_to_idx.get(element, 0)
        nodes[i, idx] = 1.0
    
    # Generate reproducible positions based on material ID
    material_id_hash = hash(structure_data['MaterialId']) % 1000
    rng = np.random.RandomState(material_id_hash)
    positions = rng.uniform(0, 5, size=(n_atoms, 3))
    
    # Create box (unit cell)
    box = np.eye(3) * 5.0
    
    # Create edges - each atom connects to 3 neighbors
    senders = []
    receivers = []
    
    for i in range(n_atoms):
        # Choose 3 random neighbors for each atom
        neighbors = rng.choice([j for j in range(n_atoms) if j != i], 
                               size=min(3, n_atoms-1), 
                               replace=False)
        for j in neighbors:
            senders.append(i)
            receivers.append(j)
    
    # Create edge features (distances between atoms)
    edges = np.array([positions[r] - positions[s] for s, r in zip(senders, receivers)])
    
    # Create graph
    graph = jraph.GraphsTuple(
        nodes=jnp.array(nodes),
        edges=jnp.array(edges),
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        globals=jnp.zeros((1, 1)),  # Add a dummy global feature
        n_node=jnp.array([n_atoms]),
        n_edge=jnp.array([len(senders)])
    )
    
    return graph, jnp.array(positions), jnp.array(box), jnp.array(structure_data.get('Formation Energy Per Atom', 0.0))


def train_simple_model(structures_df, model_dir=MODEL_DIR, num_epochs=TRAINING_EPOCHS):
    """Train a simplified model for crystal energy prediction."""
    logger.info(f"Training a simplified energy prediction model for {num_epochs} epochs...")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Prepare training data
    X_train = []
    y_train = []
    
    # For each structure, extract features and energy
    for i, structure in enumerate(structures_df.iterrows()):
        structure_data = structure[1]
        try:
            graph, positions, box, energy = create_graph_from_structure(structure_data)
            
            # Extract simple features: element types and distances
            element_indices = jnp.argmax(graph.nodes, axis=1)
            
            # Calculate distances between atoms
            n_atoms = positions.shape[0]
            distances = []
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    distances.append(jnp.sqrt(jnp.sum((positions[i] - positions[j])**2)))
            
            # Features: average element index and average distance
            avg_element = jnp.mean(element_indices)
            avg_distance = jnp.mean(jnp.array(distances)) if distances else 0.0
            max_element = jnp.max(element_indices)
            min_element = jnp.min(element_indices)
            num_edges = graph.edges.shape[0]
            
            # Create feature vector
            features = jnp.array([avg_element, avg_distance, max_element, min_element, num_edges])
            
            X_train.append(features)
            y_train.append(energy)
        except Exception as e:
            logger.warning(f"Error processing structure {structure_data['MaterialId']}: {e}")
    
    # Convert to arrays
    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train)
    
    # Check if we have any training data
    if len(X_train) == 0:
        logger.error("No valid structures for training. Cannot train model.")
        return None
    
    # Normalize features
    X_mean = jnp.mean(X_train, axis=0)
    X_std = jnp.std(X_train, axis=0)
    X_std = jnp.where(X_std == 0, 1.0, X_std)  # Avoid division by zero
    X_train_norm = (X_train - X_mean) / X_std
    
    # Initialize weights randomly
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    
    # Simple linear model: E = w * X + b
    w = random.normal(subkey, (X_train_norm.shape[1], 1))
    key, subkey = random.split(key)
    b = random.normal(subkey, (1,))
    
    # Define model
    @jit
    def predict(w, b, X):
        return jnp.dot(X, w) + b
    
    # Define loss function
    @jit
    def loss_fn(w, b, X, y):
        preds = predict(w, b, X)
        return jnp.mean((preds - y[:, None]) ** 2)
    
    # Define gradient function
    grad_fn = jit(value_and_grad(loss_fn, argnums=(0, 1)))
    
    # Training loop
    losses = []
    learning_rates = []
    
    # Initial learning rate
    learning_rate = 0.01
    min_learning_rate = 1e-6
    
    # Track best loss and patience for learning rate reduction
    best_loss = float('inf')
    patience = 5
    wait = 0
    
    for epoch in range(num_epochs):
        # Compute loss and gradients
        loss_val, (grad_w, grad_b) = grad_fn(w, b, X_train_norm, y_train)
        
        # Update weights
        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b
        
        # Save loss
        losses.append(float(loss_val))
        learning_rates.append(learning_rate)
        
        # Check for learning rate reduction
        if loss_val < best_loss:
            best_loss = loss_val
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                # Reduce learning rate
                learning_rate = max(learning_rate * 0.5, min_learning_rate)
                wait = 0
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Reducing learning rate to {learning_rate:.6f}")
        
        # Log progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_val:.6f}, LR: {learning_rate:.6f}")
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(learning_rates)
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_progress.png'))
    plt.close()
    
    # Save model parameters
    model_params = {
        'w': w.tolist(),
        'b': float(b[0]),
        'X_mean': X_mean.tolist(),
        'X_std': X_std.tolist(),
        'num_epochs': num_epochs,
        'final_loss': float(losses[-1]),
        'final_lr': float(learning_rates[-1]),
    }
    
    with open(os.path.join(model_dir, 'model_params.json'), 'w') as f:
        json.dump(model_params, f, indent=2)
    
    # Save model config
    model_config = {
        'model_type': 'linear',
        'input_features': int(X_train.shape[1]),
        'output_features': 1,
        'num_epochs': num_epochs,
        'final_loss': float(losses[-1]),
        'description': 'Simple linear model trained on crystal features'
    }
    
    with open(os.path.join(model_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)
    
    logger.info(f"Model training complete. Model saved to {model_dir}")
    
    # Return the trained model and parameters
    def model_fn(graph, positions, box, box_perturbation=None):
        # Extract features from the graph and positions
        element_indices = jnp.argmax(graph.nodes, axis=1)
        
        # Calculate distances between atoms
        n_atoms = positions.shape[0]
        distances = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                distances.append(jnp.sqrt(jnp.sum((positions[i] - positions[j])**2)))
        
        # Features: average element index and average distance
        avg_element = jnp.mean(element_indices)
        avg_distance = jnp.mean(jnp.array(distances)) if distances else 0.0
        max_element = jnp.max(element_indices)
        min_element = jnp.min(element_indices)
        num_edges = graph.edges.shape[0]
        
        # Create feature vector
        features = jnp.array([avg_element, avg_distance, max_element, min_element, num_edges])
        
        # Normalize features
        features_norm = (features - jnp.array(model_params['X_mean'])) / jnp.array(model_params['X_std'])
        
        # Predict energy
        energy = jnp.dot(features_norm, jnp.array(model_params['w'])) + model_params['b']
        
        return energy[0]  # Return scalar energy
    
    return model_fn, model_params


def setup_environment():
    """Set up the environment for XAI analysis."""
    logger.info("Setting up environment for GNoME XAI analysis...")
    
    # 1. Download GNoME data
    download_data(DATA_DIR)
    
    # 2. Load structure data
    try:
        structures_df = load_structure_data(DATA_DIR, limit=100)
        
        # 3. Train a simplified model
        if len(structures_df) > 0:
            try:
                logger.info(f"Training model with {len(structures_df)} structures...")
                model_fn, model_params = train_simple_model(
                    structures_df, 
                    MODEL_DIR, 
                    num_epochs=TRAINING_EPOCHS
                )
                logger.info("Model training complete.")
            except Exception as e:
                logger.error(f"Error training model: {str(e)}")
                logger.error("Creating a simplified mock model configuration instead.")
                
                # Create model directory if it doesn't exist
                os.makedirs(MODEL_DIR, exist_ok=True)
                
                # Save a simple model config
                model_config = {
                    'model_type': 'mock',
                    'input_features': 5,
                    'output_features': 1,
                    'description': 'Mock model for crystal energy prediction'
                }
                
                with open(os.path.join(MODEL_DIR, 'model_config.json'), 'w') as f:
                    json.dump(model_config, f, indent=2)
        else:
            logger.error("No structures loaded. Cannot train model.")
    except Exception as e:
        logger.error(f"Error setting up environment: {str(e)}")
        raise
    
    logger.info("Environment setup complete. Ready for XAI analysis!")


if __name__ == "__main__":
    setup_environment()