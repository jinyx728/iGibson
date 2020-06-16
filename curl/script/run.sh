CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name handdrawer \
    --task_name pull \
    --encoder_type pixel \
    --save_tb --save_model --save_video --pre_transform_image_size 128 --image_size 84 \
    --work_dir ./tmp \
    --agent curl_sac --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 10000000 \
    --replay_buffer_capacity 200000 \
    --critic_tau 0.05 --critic_target_update_freq 1 
