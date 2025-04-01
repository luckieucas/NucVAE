# NucVAE

This repository contains the code and tools for training and evaluating a VAE-based model on nuclear instance segmentation tasks.

---

## Environment Setup

Follow the steps below to create and activate a new conda environment, then install the dependencies:

```bash
# Step 1: Create a new environment named 'nucvae' with Python 3.9
conda create -n nucvae python=3.9 -y

# Step 2: Activate the environment
conda activate nucvae

# Step 3: Install required packages from requirements.txt
pip install -r requirements.txt
---

## Data Preparation

To extract individual instance masks and corresponding image volumes from the dataset, run the following script:

\`\`\`bash
python data_utils/extract_all_instance.py \
  --mask_dir <path_to_mask_directory> \
  --image_dir <path_to_image_directory> \
  --output_mask_dir <path_to_save_extracted_masks> \
  --output_image_dir <path_to_save_extracted_images> \
  --target_shape 32 32 32
\`\`\`

**Arguments:**

- \`--mask_dir\`: Directory containing the instance masks  
- \`--image_dir\`: Directory containing the raw images  
- \`--output_mask_dir\`: Path to save processed instance masks  
- \`--output_image_dir\`: Path to save processed image patches  
- \`--target_shape\`: Target shape of the cropped volume (default: 32×32×32)

---

## Model Training

To start training the model, run:

\`\`\`bash
# Example command
python train.py -c configs/L9.yaml -m dual_vae
\`\`\`

Make sure to edit the config file to match your dataset and training parameters.

---
## Nuclei Error Detection

You can use the provided script to detect potential errors in nuclear instance masks by computing their log-likelihood under a pretrained VAE model.

### Script

```bash
python evaluate_loglikelihood.py \
  --input_folder <path_to_cropped_instances> \
  --model_path <path_to_pretrained_model.pth> \
  --output_txt <path_to_save_results.txt> \
  --num_samples 100 \
  --device cuda \
  --latent_dim 16 \
  --base_channel 16

---
