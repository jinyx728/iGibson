CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name handdrawer \
    --task_name pull \
    --encoder_type pixel \
    --action_repeat 8 \
    --save_tb --save_model --save_video --pre_transform_image_size 128 --image_size 84 \
    --work_dir ./tmp \
    --agent curl_sac --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 1000000 