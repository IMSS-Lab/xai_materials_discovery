"""
Explainable AI Framework for GNoME Models.

This module implements various XAI techniques to analyze and interpret
Graph Networks for Materials Exploration (GNoME) models, focusing on:

1. Analyzing model representations of chemical stability
2. Generating counterfactual explanations for material modifications
3. Providing XAI-driven feedback for crystal structure prediction
4. Benchmarking XAI techniques for materials science
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import flax
import jraph
from jax import grad, jit, vmap
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Optional, Union, Any
import pandas as pd
from functools import partial

# Import GNoME model components
from model import gnome, crystal, gnn


class GNoMEExplainer:
    """Base class for explaining GNoME model predictions."""
    
    def __init__(self, model, params):
        """Initialize the explainer with a trained GNoME model.
        
        Args:
            model: A trained GNoME model
            params: The parameters of the trained model
        """
        self.model = model
        self.params = params
        
    def explain(self, graph, positions, box):
        """Base explain method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")


class IntegratedGradientsExplainer(GNoMEExplainer):
    """Implements Integrated Gradients for GNoME models."""
    
    def __init__(self, model, params, steps=50):
        """Initialize with model and number of steps for path integral.
        
        Args:
            model: A trained GNoME model
            params: The parameters of the trained model
            steps: Number of steps for approximating the path integral
        """
        super().__init__(model, params)
        self.steps = steps
        
        # Create gradient function
        self.grad_fn = jit(grad(self._predict_energy, argnums=1))
        
    def _predict_energy(self, params, nodes):
        """Predict energy for a given set of node features."""
        # Create a new graph with the modified nodes
        graph = self.current_graph._replace(nodes=nodes)
        return self.model.apply(params, graph, self.current_positions, self.current_box)
    
    def explain(self, graph, positions, box, baseline=None):
        """Compute integrated gradients for the input graph.
        
        Args:
            graph: Input graph structure
            positions: Atomic positions
            box: Periodic box
            baseline: Baseline for integrated gradients (default: zeros)
            
        Returns:
            Feature attributions for each node in the graph
        """
        self.current_graph = graph
        self.current_positions = positions
        self.current_box = box
        
        # If no baseline is provided, use zeros
        if baseline is None:
            baseline = jnp.zeros_like(graph.nodes)
            
        # Define path from baseline to input
        alphas = jnp.linspace(0, 1, self.steps)
        path = [baseline + alpha * (graph.nodes - baseline) for alpha in alphas]
        
        # Compute gradients along the path
        gradients = [self.grad_fn(self.params, path_point) for path_point in path]
        
        # Average gradients and multiply by (input - baseline)
        avg_gradients = sum(gradients) / self.steps
        integrated_gradients = avg_gradients * (graph.nodes - baseline)
        
        return integrated_gradients


class GNNExplainer(GNoMEExplainer):
    """Adapts GNNExplainer for GNoME models."""
    
    def __init__(self, model, params, epochs=100, lr=0.01):
        """Initialize GNNExplainer with model and training parameters.
        
        Args:
            model: A trained GNoME model
            params: The parameters of the trained model
            epochs: Number of optimization epochs
            lr: Learning rate for mask optimization
        """
        super().__init__(model, params)
        self.epochs = epochs
        self.lr = lr
        
    def explain(self, graph, positions, box, target_energy=None):
        """Explain a prediction by learning masks over edges and nodes.
        
        Args:
            graph: Input graph structure
            positions: Atomic positions
            box: Periodic box
            target_energy: Target energy to explain (default: None, uses model prediction)
            
        Returns:
            node_mask: Importance scores for each node
            edge_mask: Importance scores for each edge
        """
        # If no target energy provided, use model prediction
        if target_energy is None:
            target_energy = self.model.apply(self.params, graph, positions, box)
        
        # Initialize masks with all ones
        node_mask = jnp.ones(graph.nodes.shape[0])
        edge_mask = jnp.ones(graph.edges.shape[0]) if graph.edges is not None else None
            
        # Define optimization function for the masks
        @jit
        def mask_loss(node_mask, edge_mask):
            # Apply masks to graph
            masked_nodes = graph.nodes * node_mask[:, None]
            masked_edges = None
            if edge_mask is not None and graph.edges is not None:
                masked_edges = graph.edges * edge_mask[:, None]
            
            masked_graph = graph._replace(nodes=masked_nodes, edges=masked_edges)
            
            # Get prediction with masked graph
            masked_pred = self.model.apply(self.params, masked_graph, positions, box)
            
            # Loss is MSE between original and masked predictions
            pred_loss = jnp.mean((masked_pred - target_energy) ** 2)
            
            # Add regularization for sparsity
            mask_sum = jnp.sum(node_mask)
            if edge_mask is not None:
                mask_sum += jnp.sum(edge_mask)
            
            return pred_loss + 0.01 * mask_sum
            
        # Simple gradient descent to optimize masks
        for _ in range(self.epochs):
            node_grad = grad(lambda nm: mask_loss(nm, edge_mask))(node_mask)
            node_mask = node_mask - self.lr * node_grad
            node_mask = jnp.clip(node_mask, 0, 1)  # Constrain to [0, 1]
            
            if edge_mask is not None:
                edge_grad = grad(lambda em: mask_loss(node_mask, em))(edge_mask)
                edge_mask = edge_mask - self.lr * edge_grad
                edge_mask = jnp.clip(edge_mask, 0, 1)  # Constrain to [0, 1]
                
        return node_mask, edge_mask


class CounterfactualExplainer(GNoMEExplainer):
    """Generates counterfactual explanations for GNoME predictions."""
    
    def __init__(self, model, params, elements_list, optimization_steps=100, lr=0.001):
        """Initialize counterfactual explainer.
        
        Args:
            model: A trained GNoME model
            params: The parameters of the trained model
            elements_list: List of possible elements to substitute
            optimization_steps: Number of optimization steps
            lr: Learning rate for optimization
        """
        super().__init__(model, params)
        self.elements_list = elements_list
        self.optimization_steps = optimization_steps
        self.lr = lr
        
    def explain(self, graph, positions, box, target_stability=0.0, 
                perturbation_type='substitution', site_indices=None):
        """Generate counterfactual explanations.
        
        Args:
            graph: Input graph structure
            positions: Atomic positions
            box: Periodic box
            target_stability: Target stability value to achieve
            perturbation_type: Type of perturbation ('substitution', 'vacancy', 'distortion')
            site_indices: Specific sites to consider for perturbation (default: all)
            
        Returns:
            A dict containing counterfactual explanations and metrics
        """
        original_energy = self.model.apply(self.params, graph, positions, box)
        
        if perturbation_type == 'substitution':
            return self._substitution_counterfactual(
                graph, positions, box, original_energy, target_stability, site_indices)
        elif perturbation_type == 'vacancy':
            return self._vacancy_counterfactual(
                graph, positions, box, original_energy, target_stability, site_indices)
        elif perturbation_type == 'distortion':
            return self._distortion_counterfactual(
                graph, positions, box, original_energy, target_stability, site_indices)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
    
    def _substitution_counterfactual(self, graph, positions, box, 
                                   original_energy, target_stability, site_indices):
        """Generate counterfactual by element substitution."""
        # If no site indices provided, consider all sites
        if site_indices is None:
            site_indices = range(graph.nodes.shape[0])
            
        counterfactuals = []
        
        # For each site, try substituting with each element
        for site_idx in site_indices:
            original_element = np.argmax(graph.nodes[site_idx])
            
            for new_element_idx, element in enumerate(self.elements_list):
                if new_element_idx == original_element:
                    continue
                    
                # Create new node features with substitution
                new_nodes = graph.nodes.copy()
                new_nodes = new_nodes.at[site_idx].set(jnp.zeros_like(new_nodes[site_idx]))
                new_nodes = new_nodes.at[site_idx, new_element_idx].set(1.0)
                
                # Create new graph with substitution
                new_graph = graph._replace(nodes=new_nodes)
                
                # Predict energy with new composition
                new_energy = self.model.apply(self.params, new_graph, positions, box)
                
                # Calculate energy difference
                energy_diff = new_energy - original_energy
                
                counterfactuals.append({
                    'site_index': site_idx,
                    'original_element': self.elements_list[original_element],
                    'new_element': element,
                    'energy_difference': energy_diff.item(),
                    'new_energy': new_energy.item(),
                    'distance_to_target': abs(new_energy - target_stability).item()
                })
        
        # Sort by distance to target stability
        counterfactuals.sort(key=lambda x: x['distance_to_target'])
        
        return {
            'counterfactuals': counterfactuals,
            'original_energy': original_energy.item(),
            'target_stability': target_stability
        }
    
    def _vacancy_counterfactual(self, graph, positions, box, 
                              original_energy, target_stability, site_indices):
        """Generate counterfactual by creating vacancies."""
        # Implementation would remove nodes/atoms from the structure
        # This is more complex as it requires creating a new graph structure
        # Simplified implementation for demonstration
        if site_indices is None:
            site_indices = range(graph.nodes.shape[0])
            
        counterfactuals = []
        
        # For each site, simulate a vacancy by zeroing out node features
        for site_idx in site_indices:
            original_element = np.argmax(graph.nodes[site_idx])
            
            # Create new node features with vacancy
            new_nodes = graph.nodes.copy()
            new_nodes = new_nodes.at[site_idx].set(jnp.zeros_like(new_nodes[site_idx]))
            
            # Create new graph with vacancy
            new_graph = graph._replace(nodes=new_nodes)
            
            # Predict energy with vacancy
            new_energy = self.model.apply(self.params, new_graph, positions, box)
            
            # Calculate energy difference
            energy_diff = new_energy - original_energy
            
            counterfactuals.append({
                'site_index': site_idx,
                'original_element': self.elements_list[original_element],
                'action': 'create_vacancy',
                'energy_difference': energy_diff.item(),
                'new_energy': new_energy.item(),
                'distance_to_target': abs(new_energy - target_stability).item()
            })
        
        # Sort by distance to target stability
        counterfactuals.sort(key=lambda x: x['distance_to_target'])
        
        return {
            'counterfactuals': counterfactuals,
            'original_energy': original_energy.item(),
            'target_stability': target_stability
        }
    
    def _distortion_counterfactual(self, graph, positions, box, 
                                 original_energy, target_stability, site_indices):
        """Generate counterfactual by atomic position distortion."""
        # If no site indices provided, consider all sites
        if site_indices is None:
            site_indices = range(positions.shape[0])
            
        counterfactuals = []
        
        # For each site, apply small distortions
        for site_idx in site_indices:
            original_element = np.argmax(graph.nodes[site_idx])
            
            # Try different distortion magnitudes
            for magnitude in [0.01, 0.05, 0.1, 0.2]:
                # Distort in +x direction
                new_positions = positions.copy()
                new_positions = new_positions.at[site_idx, 0].add(magnitude)
                
                # Predict energy with distorted position
                new_energy = self.model.apply(self.params, graph, new_positions, box)
                
                # Calculate energy difference
                energy_diff = new_energy - original_energy
                
                counterfactuals.append({
                    'site_index': site_idx,
                    'original_element': self.elements_list[original_element],
                    'action': f'distort_+x_{magnitude}',
                    'energy_difference': energy_diff.item(),
                    'new_energy': new_energy.item(),
                    'distance_to_target': abs(new_energy - target_stability).item()
                })
                
                # Distort in -x direction
                new_positions = positions.copy()
                new_positions = new_positions.at[site_idx, 0].add(-magnitude)
                
                # Predict energy with distorted position
                new_energy = self.model.apply(self.params, graph, new_positions, box)
                
                # Calculate energy difference
                energy_diff = new_energy - original_energy
                
                counterfactuals.append({
                    'site_index': site_idx,
                    'original_element': self.elements_list[original_element],
                    'action': f'distort_-x_{magnitude}',
                    'energy_difference': energy_diff.item(),
                    'new_energy': new_energy.item(),
                    'distance_to_target': abs(new_energy - target_stability).item()
                })
                
                # Could also distort in y and z, but omitted for brevity
        
        # Sort by distance to target stability
        counterfactuals.sort(key=lambda x: x['distance_to_target'])
        
        return {
            'counterfactuals': counterfactuals,
            'original_energy': original_energy.item(),
            'target_stability': target_stability
        }


class XAIBenchmark:
    """Benchmarks and evaluates different XAI techniques."""
    
    def __init__(self, explainers, evaluation_metrics=None):
        """Initialize with a list of explainers and evaluation metrics.
        
        Args:
            explainers: Dict mapping explainer names to explainer instances
            evaluation_metrics: Dict mapping metric names to metric functions
        """
        self.explainers = explainers
        
        # Default evaluation metrics if none provided
        if evaluation_metrics is None:
            self.evaluation_metrics = {
                'fidelity': self._fidelity_metric,
                'sparsity': self._sparsity_metric,
                'stability': self._stability_metric
            }
        else:
            self.evaluation_metrics = evaluation_metrics
    
    def benchmark(self, graph, positions, box, ground_truth=None):
        """Run benchmarks for all explainers on the given input.
        
        Args:
            graph: Input graph structure
            positions: Atomic positions
            box: Periodic box
            ground_truth: Optional ground truth explanations for comparison
            
        Returns:
            Dict of benchmark results for each explainer and metric
        """
        results = {}
        
        # Run each explainer and evaluate metrics
        for name, explainer in self.explainers.items():
            explanation = explainer.explain(graph, positions, box)
            results[name] = {}
            
            for metric_name, metric_fn in self.evaluation_metrics.items():
                results[name][metric_name] = metric_fn(explanation, graph, positions, box, ground_truth)
            
        return results
    
    def _fidelity_metric(self, explanation, graph, positions, box, ground_truth=None):
        """Measure how well the explanation predicts model behavior.
        
        For IntegratedGradients and GNNExplainer, this might measure how well
        masking unimportant features preserves model output.
        
        For CounterfactualExplainer, this might measure how close the counterfactual
        gets to the target stability.
        """
        # Implementation depends on the explainer type
        if isinstance(explanation, tuple) and len(explanation) == 2:
            # Likely GNNExplainer output (node_mask, edge_mask)
            node_mask, edge_mask = explanation
            fidelity = np.mean(node_mask)  # Simplified for demonstration
            return fidelity
        elif isinstance(explanation, dict) and 'counterfactuals' in explanation:
            # Likely CounterfactualExplainer output
            if explanation['counterfactuals']:
                # Return how close the best counterfactual gets to target
                return explanation['counterfactuals'][0]['distance_to_target']
            else:
                return float('inf')
        else:
            # Likely IntegratedGradientsExplainer output
            return np.mean(np.abs(explanation))
    
    def _sparsity_metric(self, explanation, graph, positions, box, ground_truth=None):
        """Measure how sparse/simple the explanation is."""
        # Implementation depends on the explainer type
        if isinstance(explanation, tuple) and len(explanation) == 2:
            # Likely GNNExplainer output (node_mask, edge_mask)
            node_mask, edge_mask = explanation
            # Measure percentage of features with importance > 0.5
            sparsity = np.mean(node_mask > 0.5)
            return sparsity
        elif isinstance(explanation, dict) and 'counterfactuals' in explanation:
            # Likely CounterfactualExplainer output
            # Return number of counterfactuals needed to reach target
            return len([cf for cf in explanation['counterfactuals'] 
                       if cf['distance_to_target'] < 0.05])
        else:
            # Likely IntegratedGradientsExplainer output
            return np.mean(np.abs(explanation) > 0.1)
    
    def _stability_metric(self, explanation, graph, positions, box, ground_truth=None):
        """Measure how stable the explanation is to small input perturbations."""
        # This would require running multiple explanations with perturbed inputs
        # Simplified for demonstration
        return 0.95  # Placeholder


class XAIVisualizer:
    """Visualizes XAI explanations for GNoME models."""
    
    def __init__(self, elements_list):
        """Initialize visualizer with a list of elements."""
        self.elements_list = elements_list
    
    def visualize_integrated_gradients(self, graph, integrated_gradients, title="Integrated Gradients"):
        """Visualize integrated gradients as a bar chart for each node/atom."""
        n_nodes = graph.nodes.shape[0]
        
        # Sum gradients across features for each node
        node_importance = np.sum(np.abs(integrated_gradients), axis=1)
        
        # Get element types
        element_types = [self.elements_list[np.argmax(graph.nodes[i])] for i in range(n_nodes)]
        
        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(n_nodes), node_importance)
        
        # Color bars by element
        unique_elements = list(set(element_types))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_elements)))
        color_map = {elem: color for elem, color in zip(unique_elements, colors)}
        
        for i, bar in enumerate(bars):
            bar.set_color(color_map[element_types[i]])
        
        plt.xlabel('Atom Index')
        plt.ylabel('Feature Importance')
        plt.title(title)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[elem], label=elem) 
                          for elem in unique_elements]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_gnn_explainer(self, graph, node_mask, edge_mask=None, title="GNN Explainer"):
        """Visualize node and edge masks from GNN Explainer."""
        n_nodes = graph.nodes.shape[0]
        
        # Get element types
        element_types = [self.elements_list[np.argmax(graph.nodes[i])] for i in range(n_nodes)]
        
        # Create bar chart for node importances
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(n_nodes), node_mask)
        
        # Color bars by element
        unique_elements = list(set(element_types))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_elements)))
        color_map = {elem: color for elem, color in zip(unique_elements, colors)}
        
        for i, bar in enumerate(bars):
            bar.set_color(color_map[element_types[i]])
        
        plt.xlabel('Atom Index')
        plt.ylabel('Node Importance')
        plt.title(title)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[elem], label=elem) 
                          for elem in unique_elements]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_counterfactuals(self, counterfactual_results, top_n=5, 
                                title="Top Counterfactual Modifications"):
        """Visualize top counterfactual explanations."""
        if not counterfactual_results.get('counterfactuals'):
            print("No counterfactuals available to visualize")
            return None
        
        # Get top N counterfactuals
        top_cfs = counterfactual_results['counterfactuals'][:top_n]
        
        # Create bar chart for energy differences
        plt.figure(figsize=(12, 6))
        
        labels = []
        energy_diffs = []
        colors = []
        
        for i, cf in enumerate(top_cfs):
            if 'new_element' in cf:
                # Element substitution
                label = f"{cf['original_element']} → {cf['new_element']} at site {cf['site_index']}"
            elif 'action' in cf and cf['action'].startswith('distort'):
                # Distortion
                label = f"{cf['action']} at site {cf['site_index']}"
            else:
                # Vacancy
                label = f"Vacancy at site {cf['site_index']} ({cf['original_element']})"
                
            labels.append(label)
            energy_diffs.append(cf['energy_difference'])
            
            # Red for destabilizing, green for stabilizing
            colors.append('green' if energy_diffs[-1] < 0 else 'red')
        
        bars = plt.bar(range(len(labels)), energy_diffs, color=colors)
        
        plt.xlabel('Modification')
        plt.ylabel('Energy Difference (eV)')
        plt.title(title)
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        
        # Add reference line at zero
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add original and target energy information
        plt.figtext(0.01, 0.01, 
                   f"Original Energy: {counterfactual_results['original_energy']:.4f} eV\n"
                   f"Target Stability: {counterfactual_results['target_stability']:.4f} eV", 
                   va="bottom", ha="left")
        
        plt.tight_layout()
        return plt.gcf()