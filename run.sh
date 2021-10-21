python3 main.py \
    --dataset='svhn' \
    --datapath='./data/' \
    --lr=0.001 \
    --training_phase='first' \
    --ccebal=1 \
    --batch=512 \
    --batchv=512 \
    --T=7 \
    --logfolder='./svhn_log_first' \
    --epochs=1 \
    --pretrain_checkpoint=None

python3 main.py \
    --dataset='svhn' \
    --datapath='./data/' \
    --lr=0.001 \
    --training_phase='second' \
    --ccebal=0 \
    --batch=512 \
    --batchv=512 \
    --T=7 \
    --logfolder='./svhn_log_second' \
    --epochs=1 \
    --pretrain_checkpoint='./svhn_log_first/weights_f_0.pth'

python3 main.py \
    --dataset='svhn' \
    --datapath='./data/' \
    --lr=0.001 \
    --training_phase='third' \
    --ccebal=16 \
    --batch=64 \
    --batchv=64 \
    --T=7 \
    --logfolder='./svhn_log_third' \
    --epochs=1 \
    --pretrain_checkpoint='./svhn_log_second/weights_f_0.pth'
