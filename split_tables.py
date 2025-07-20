import pandas as pd

# load the full results
df = pd.read_csv('compression_results.csv')

# mapping of dataset names to output files
datasets = {
    'enwik8': 'text_results.csv',
    'imagenet': 'image_results.csv',
    'librispeech': 'speech_results.csv'
}

for ds, out in datasets.items():
    sub = df[df['dataset'] == ds][['model', 'checkpoint', 'compression_ratio']].copy()
    sub['compression_ratio'] = sub['compression_ratio'].round(3)
    sub = sub.rename(columns={'checkpoint': 'training_step'})
    sub.to_csv(out, index=False)

