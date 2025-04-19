
source ~/anaconda3/etc/profile.d/conda.sh
conda activate forgery-watermark
export CUDA_VISIBLE_DEVICES=7

nohup python sample.py --output_dir /work/forgery/Data/StableSignature --num_samples 7000 --batch_size 4 > sample.log & 
