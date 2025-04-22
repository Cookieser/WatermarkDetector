
source ~/anaconda3/etc/profile.d/conda.sh
conda activate forgery-watermark


# export CUDA_VISIBLE_DEVICES=7
# nohup python sample.py --output_dir /work/forgery/Data/StableSignature \
#  --id_range_start 4268 \
#  --id_range_end 5767 \
#  --batch_size 4 > sample.log & 

# export CUDA_VISIBLE_DEVICES=2
#  nohup python sample.py --output_dir /work/forgery/Data/StableSignature2 \
#  --id_range_start 5768 \
#  --id_range_end 7499 \
#  --batch_size 4 > sample2.log & 


export CUDA_VISIBLE_DEVICES=2
 nohup python sample.py --output_dir /work/forgery/Data/StableSignature \
 --id_range_start 7499 \
 --id_range_end 7500 \
 --batch_size 1 > sample.log & 