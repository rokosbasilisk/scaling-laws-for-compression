#!/usr/bin/env python3
"""
Generate LaTeX tables from compression_results.csv
This script reads the CSV file and generates three separate .tex files
that can be included in the main LaTeX document.
"""

import pandas as pd
import sys

def generate_latex_tables():
    """Generate LaTeX table files from compression_results.csv"""
    
    try:
        # Read the CSV file
        df = pd.read_csv('compression_results.csv')
        print(f"Loaded {len(df)} rows from compression_results.csv")
        
        # Print column names to debug
        print("Columns:", list(df.columns))
        
        # Generate table for each modality
        modalities = {
            'enwik8': ('text_table.tex', 'Text compression results on Enwik8 dataset'),
            'imagenet': ('image_table.tex', 'Image compression results on ImageNet patches'),
            'librispeech': ('speech_table.tex', 'Speech compression results on LibriSpeech')
        }
        
        for dataset, (filename, caption) in modalities.items():
            # Filter data for this modality
            subset = df[df['dataset'] == dataset].copy()
            
            if len(subset) == 0:
                print(f"Warning: No data found for dataset '{dataset}'")
                continue
                
            print(f"Generating {filename} with {len(subset)} rows")
            
            # Sort by model size (extract number from model name) and checkpoint
            def extract_model_size(model_name):
                """Extract model size in millions from model name like 'pythia-70m'"""
                if 'pythia-' in model_name:
                    size_str = model_name.split('pythia-')[1].rstrip('mb')
                    if size_str.endswith('b'):
                        return float(size_str[:-1]) * 1000  # Convert billions to millions
                    elif size_str.endswith('m'):
                        return float(size_str[:-1])
                    else:
                        return float(size_str)
                return 0
            
            def extract_checkpoint_num(checkpoint):
                """Extract checkpoint number from string like 'step1000'"""
                return int(checkpoint.replace('step', ''))
            
            subset['model_size'] = subset['model'].apply(extract_model_size)
            subset['checkpoint_num'] = subset['checkpoint'].apply(extract_checkpoint_num)
            subset = subset.sort_values(['model_size', 'checkpoint_num'])
            
            # Generate LaTeX table content
            latex_content = []
            
            for _, row in subset.iterrows():
                model = row['model']
                checkpoint = row['checkpoint'] 
                cr = row['compression_ratio']
                
                # Format compression ratio to 2 decimal places
                cr_formatted = f"{cr:.2f}"
                
                latex_content.append(f"{model} & {checkpoint} & {cr_formatted} \\\\")
            
            # Write to file
            with open(filename, 'w') as f:
                f.write('\n'.join(latex_content))
            
            print(f"Generated {filename}")
        
        print("All table files generated successfully!")
        print("\nTo use these in your LaTeX document, replace the table contents with:")
        print("\\input{text_table.tex}")
        print("\\input{image_table.tex}")  
        print("\\input{speech_table.tex}")
        
    except FileNotFoundError:
        print("Error: compression_results.csv not found in current directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_latex_tables()
