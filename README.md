# DL-Final-Competition


Downloading subset of Open Images train Dataset

```bash
cd data/Open_Images
curl -L -O https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv
curl -L -O https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv
curl -L -O https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv

# Run all cells in the notebook/open_images_analysis.ipynb from the PROJECT ROOT

# cd to the POJECT ROOT and run the following commands
# sample size should be a multiple of 1000
python scripts/sample_open_images.py --sample-size=200000

python scripts/download_open_images.py data/Open_Images/train_sampled_200k.txt  --download_folder=data/Open_Images/train/raw --num_processes=5

# or on HPC Cluster
sbatch sbatch/download_open_images.slurm
```
