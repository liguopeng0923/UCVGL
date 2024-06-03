# Semi-Supervised Curriculum Learning
python -u train_crossview_sat.py --lr 0.0001 --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 40 --op adamw --dataset vigor --data-folder "./data/VIGOR" --batch-size 128 --workers=16 --gt-ratio 0.1 --sat-size 384 384 --grd-size 384 768 --threshold 0.035

# Evaluation
# python -u train_crossview_sat.py --lr 0.0001 --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 40 --op adamw --dataset vigor --data-folder "./data/VIGOR" --batch-size 64 --workers=16 --gt-ratio 0.1 --sat-size 384 384 --grd-size 384 768 --threshold 0.035 --checkpoint ./ckpt/vigor/train_crossviewsat.pth.tar --evaluate