deepspeed --include localhost:0,1,2,3 \
    dist_train.py --with_cuda --use_ema \
    --deepspeed --deepspeed_config ds_config.json