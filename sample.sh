
source ~/anaconda3/etc/profile.d/conda.sh
conda activate forgery-watermark
export CUDA_VISIBLE_DEVICES=7

nohup python sample.py --output_dir test --num_samples 32 --batch_size 4 > sample.log & 
