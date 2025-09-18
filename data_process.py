#!/usr/bin/env python3
"""
Data Processor for Surgical Expertise Classification
Converts full_data.json to PKL format compatible with original_SC/test.py
"""

import json
import pickle
import numpy as np
import warnings
from typing import List, Tuple, Dict, Any
import os

class SurgicalDataProcessor:
    def __init__(self):
        """Initialize the surgical data processor"""
        self.expertise_mapping = {
            'Medical student': 0,
            'Resident PGY1': 1,
            'Resident PGY2': 2,
            'Resident PGY3': 3,
            'Resident PGY4': 4,
            'Resident PGY5': 5,
            'Resident PGY6': 6,
            'Fellow': 7,
            'Staff': 8
        }
        
    def load_json_data(self, json_path: str) -> Dict[str, Any]:
        """Load data from JSON file"""
        print(f"📂 Loading data from {json_path}")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File not found: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        print(f"✅ Loaded data with {len(data)} main keys")
        return data
    
    def convert_to_list_format(self, json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert JSON data to list format expected by normalization method"""
        print("🔄 Converting JSON data to list format...")
        
        all_data = []
        
        # Get the data dimensions
        participant_data = json_data.get('participant', {})
        trial_data = json_data.get('trial', {})
        expertise_data = json_data.get('expertise', {})
        level_data = json_data.get('level', {})
        instrument_data = json_data.get('instrument', {})
        metric_data = json_data.get('metric', {})
        data_matrices = json_data.get('data', {})
        
        # Get all unique sample indices
        sample_indices = sorted([int(idx) for idx in data_matrices.keys()])
        print(f"📊 Found {len(sample_indices)} data points")
        
        # Treat each data point as a separate sample
        for idx in sample_indices:
            str_idx = str(idx)
            
            # Skip if essential data is missing
            if (str_idx not in data_matrices or 
                str_idx not in expertise_data or 
                str_idx not in level_data):
                continue
                
            # Get sample metadata
            participant = participant_data.get(str_idx, f"P{idx}")
            trial = trial_data.get(str_idx, f"T{idx}")
            expertise = expertise_data.get(str_idx, "Medical student")
            level = level_data.get(str_idx, "Medical student")
            instrument = instrument_data.get(str_idx, "unknown")
            metric = metric_data.get(str_idx, "unknown")
            data_vector = data_matrices.get(str_idx, [])
            
            if not data_vector:
                continue
                
            # Convert data to numpy array and reshape to 2D (1 metric, n_timepoints)
            data_array = np.array(data_vector).reshape(1, -1)
            
            # Add expertise level as first row (as expected by normalization)
            expertise_level = self.expertise_mapping.get(level, 0)
            expertise_row = np.full((1, data_array.shape[1]), expertise_level)
            final_matrix = np.vstack([expertise_row, data_array])
            
            # Create entry in expected format
            entry = {
                'name': f"{participant}_trial_{trial}_metric_{metric}_idx_{idx}",
                'data': final_matrix,
                'level': level,
                'expertise': expertise,
                'participant': participant,
                'trial': trial,
                'metric': metric,
                'idx': idx
            }
            
            all_data.append(entry)
        
        print(f"✅ Converted {len(all_data)} valid samples")
        if all_data:
            print(f"📏 Example data matrix shape: {all_data[0]['data'].shape}")
        return all_data
    
    def _normalize_data(self, all_data: List, normalization_data: List) -> Tuple[List, List]:
        """
        Normalize data using z-score normalization 
        Modified to handle variable length sequences by grouping by length
        
        Args:
            all_data: All processed data (list of dictionaries with level info)
            normalization_data: Data to use for computing normalization parameters
            
        Returns:
            Tuple of (normalized_data, original_data)
        """
        if not normalization_data:
            warnings.warn("No normalization data available, using all data for normalization")
            normalization_data = [entry['data'] for entry in all_data]
        
        # Group data by shape for normalization
        shape_groups = {}
        for i, entry in enumerate(all_data):
            data_shape = entry['data'].shape
            if data_shape not in shape_groups:
                shape_groups[data_shape] = []
            shape_groups[data_shape].append((i, entry))
        
        print(f"📊 Found {len(shape_groups)} different data shapes")
        for shape, entries in shape_groups.items():
            print(f"   Shape {shape}: {len(entries)} samples")
        
        final_data = []
        final_data_actual = []
        
        # Process each shape group separately
        for data_shape, entries in shape_groups.items():
            indices, group_entries = zip(*entries)
            group_data = [entry['data'] for entry in group_entries]
            
            # Concatenate normalization data for this shape group
            concat_norm_data = np.concatenate(group_data, axis=1)
            
            # Compute mean and std (excluding expertise row)
            metrics_mean = np.mean(concat_norm_data[1:, :], axis=1, keepdims=True)
            metrics_std = np.std(concat_norm_data[1:, :], axis=1, keepdims=True)
            
            # Add small epsilon to avoid division by zero
            metrics_std = metrics_std + 1e-8
            
            # Normalize all data in this group
            for entry in group_entries:
                data = entry['data']
                
                # Keep original data with level information
                actual_entry = {
                    'name': entry['name'],
                    'data': data.copy(),
                    'level': entry['level'],
                    'expertise': entry['expertise'],
                    'participant': entry['participant'],
                    'trial': entry['trial']
                }
                final_data_actual.append(actual_entry)
                
                # Normalize data (skip expertise row)
                normalized_data = data.copy()
                normalized_data[1:, :] = (data[1:, :] - metrics_mean) / metrics_std
                
                # Create normalized entry with level information
                normalized_entry = {
                    'name': entry['name'],
                    'data': normalized_data,
                    'level': entry['level'],
                    'expertise': entry['expertise'],
                    'participant': entry['participant'],
                    'trial': entry['trial']
                }
                final_data.append(normalized_entry)
        
        return final_data, final_data_actual
    
    def process_and_save(self, json_path: str, output_path: str):
        """Complete processing pipeline"""
        print("🚀 Starting data processing pipeline...")
        
        # Load JSON data
        json_data = self.load_json_data(json_path)
        
        # Convert to list format
        all_data = self.convert_to_list_format(json_data)
        
        if not all_data:
            raise ValueError("No valid data found after conversion")
        
        # Use all data for normalization (as in original method)
        normalization_data = [entry['data'] for entry in all_data]
        
        # Apply exact normalization method
        print("🔧 Applying exact normalization method...")
        normalized_data, original_data = self._normalize_data(all_data, normalization_data)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save normalized data (this is what test.py expects)
        print(f"💾 Saving normalized data to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(normalized_data, f)
        
        # Also save original data for reference
        original_path = output_path.replace('.pkl', '_original.pkl')
        print(f"💾 Saving original data to {original_path}")
        with open(original_path, 'wb') as f:
            pickle.dump(original_data, f)
        
        # Print summary
        print(f"\n📊 Processing Summary:")
        print(f"   - Total samples processed: {len(normalized_data)}")
        print(f"   - Data shape per sample: {normalized_data[0]['data'].shape if normalized_data else 'N/A'}")
        print(f"   - Expertise levels found: {set(entry['level'] for entry in normalized_data)}")
        print(f"   - Output files:")
        print(f"     * Normalized: {output_path}")
        print(f"     * Original: {original_path}")
        print("✅ Processing completed successfully!")

def main():
    """Main execution function"""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'data', 'full_data.json')
    output_path = os.path.join(script_dir, 'data', 'final_data_normalized_with_levels.pkl')
    
    # Create processor and run
    processor = SurgicalDataProcessor()
    
    try:
        processor.process_and_save(json_path, output_path)
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()