# 1. Geometric Projection
python3 CFP/GeometricProjection/CVACT.py

# 2. Image Projection
cd CFP/CycleGAN

python train.py --dataroot ../../data/CVACT/ --name CVACT --model cycle_gan

python test.py --dataroot ../../data/CVACT/ --name CVACT --model cycle_gan --results_dir ../../data/CVACT/g2a_sat

cd ../../

# 3. Intra-View Self-Supervised Learning
python -u train_intraview.py --lr 0.0001 --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 12 --op adamw --dataset cvact --data-folder "./data/CVACT" --batch-size 128 --workers=16

# 4. Cross-View Self-Supervised Learning
python -u train_crossview_fake.py --lr 0.0001 --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 28 --op adamw --dataset cvact --data-folder "./data/CVACT" --batch-size 128 --workers=16 --checkpoint ./ckpt/cvact/train_crossviewfake.pth.tar

# 5. Semi-Supervised Curriculum Learning
python -u train_crossview_sat.py --lr 0.0001 --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 40 --op adamw --dataset cvact --data-folder "./data/CVACT" --batch-size 128 --workers=16 --checkpoint ./ckpt/cvact/train_crossviewfake.pth.tar --threshold 0.025

# Evaluation
# python -u train_crossview_sat.py --lr 0.0001 --dist-url 'tcp://localhost:10005' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 40 --op adamw --dataset cvact --data-folder "./data/CVACT" --batch-size 64 --workers=16 --checkpoint ./ckpt/cvact/train_crossviewsat.pth.tar --threshold 0.05 --evaluate





