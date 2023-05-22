GPU=$2
port=23488


# config=swin-base-patch4-window7_upernet_8xb2-160k_fault-512x512
# config=swin-base-patch4-window7_upernet_8xb2-160k_fault_public-512x512
# config=swin-base-patch4-window7_upernet_8xb2-160k_fault_public-128x128
# config=swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice-512x512
# config=swin-base-patch4-window7_upernet_8xb2-16k_fault_real_labeled_mini-128x128_ft
# config=swin-base-patch4-window7_upernet_8xb2-160k_fault_real_labeled_slice-256x256_ft
# config=swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice_25d-256x256
# config=swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice_25d-512x512
# config=swin-base-patch4-window7_upernet_8xb2-160k_fault_real_labeled_slice_25d-256x256_ft
config=swin-base-patch4-window7_upernet_8xb2-160k_fault_2Dfault_0519_force3chan-256x256
if [ $1 = "train" ]; then
    CUDA_VISIBLE_DEVICES=$GPU PORT=${port} ./tools/dist_train.sh ./projects/Fault_recong/config/${config}.py 1 --work-dir output/${config} 
elif [ $1 = "test" ]; then
    CUDA_VISIBLE_DEVICES=$GPU ./tools/dist_test.sh ./output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public-128x128/swin-base-patch4-window7_upernet_8xb2-160k_fault_public-128x128.py ./output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public-128x128/iter_48000.pth 1
fi