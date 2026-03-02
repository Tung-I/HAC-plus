import os

# for lmbda in [0.004]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
#     for cuda, scene in enumerate(['truck', 'train']):
#         mask_lr_final = 0.0001 * lmbda / 0.001
#         one_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s data/tandt/{scene} --eval --lod 0 --voxel_size 0.01 --update_init_factor 16 --iterations 30_000 -m outputs/tandt/{scene}/{lmbda} --lmbda {lmbda} --mask_lr_final {mask_lr_final}'
#         os.system(one_cmd)


# The five scenes from the Tanks and Temples dataset
tnt_scenes = ['truck']

for lmbda in [0.001]:  
    for scene in tnt_scenes:
        # Note: --eval is already present here, which handles the train/test split for COLMAP
        mask_lr_final = 0.0001 * lmbda / 0.001
        
        one_cmd = (
            f'CUDA_VISIBLE_DEVICES=0 python train.py '
            f'-s data/tandt/{scene} '
            f'--eval --lod 0 --voxel_size 0.01 --white_background '
            f'--update_init_factor 16 --iterations 30_000 '
            f'-m outputs/tandt/{scene}/{lmbda} '
            f'--lmbda {lmbda} --mask_lr_final {mask_lr_final}'

        )
        
        print(f"--- Starting Training: {scene} with lambda {lmbda} ---")
        os.system(one_cmd)