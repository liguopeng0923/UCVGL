# 1. Geometric Projection
python3 CFP/GeometricProjection/CVUSA.py

# 2. Image Projection
cd CFP/CycleGAN

python train.py --dataroot ../../data/CVUSA/ --name CVUSA --model cycle_gan

python test.py --dataroot ../../data/CVUSA/ --name CVUSA --model cycle_gan --results_dir ../../data/CVUSA/g2a_sat

cd ../../

# 3. Intra-View Self-Supervised Learning
python -u train_intraview.py --lr 0.0001 --dist-url 'tcp://localhost:10000' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 12 --op adamw --dataset cvusa --data-folder "./data/CVUSA" --batch-size 128 --workers=16 

# 4. Cross-View Self-Supervised Learning
python -u train_crossview_fake.py --lr 0.0001 --dist-url 'tcp://localhost:10000' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 28 --op adamw --dataset cvusa --data-folder "./data/CVUSA" --batch-size 128 --workers=16 --checkpoint ./ckpt/cvusa/train_intraview.pth.tar

# 5. Semi-Supervised Curriculum Learning
python -u train_crossview_sat.py --lr 0.0001 --dist-url 'tcp://localhost:10000' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 40 --op adamw --dataset cvusa --data-folder "./data/CVUSA" --batch-size 128 --workers=16 --checkpoint ./ckpt/cvusa/train_crossviewfake.pth.tar --threshold 0.05

# Evaluation
#python -u train_crossview_sat.py --lr 0.0001 --dist-url 'tcp://localhost:10000' --multiprocessing-distributed --world-size 1 --rank 0  --epochs 40 --op adamw --dataset cvusa --data-folder "./data/CVUSA" --batch-size 128 --workers=16 --checkpoint ./ckpt/train_crossviewsat.pth.tar --threshold 0.05 --evaluate


