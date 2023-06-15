GPU=$2
port=23490


# config=simmim_swin-base-w6_100e_512x512_public
# config=simmim_swin-base-w7_3000e_256x256_0519_2d_fault
# config=simmim_swin-base-w7_3000e_512x512_public_25d
# config=simmim_swin-base-w7_3000e_256x256_0519_2d_fault_per_image_norm
# config=simmim_swin-base-w7_3000e_512x512_public_25d_per_image_norm
# config=simmim_swin-base-w7_3000e_512x512_public_force_3_chan_per_image_norm
# config=simmim_swin-base-w7_100e_512x512_mix_force_3_chan_per_image_norm
config=mae_vit-base-p16_8xb512-amp-coslr-300e_mix_force_3_chan_per_image_norm_norm_pix_False

if [ $1 = "train" ]; then
    CUDA_VISIBLE_DEVICES=$GPU PORT=${port} ./tools/dist_train.sh ./projects/Fault_Recong/config/${config}.py 2 --work-dir output/${config} 
elif [ $1 = "test" ]; then
    CUDA_VISIBLE_DEVICES=$GPU ./tools/dist_test.sh ./output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public-128x128/swin-base-patch4-window7_upernet_8xb2-160k_fault_public-128x128.py ./output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public-128x128/iter_48000.pth 1
fi