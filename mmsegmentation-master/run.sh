GPU=$2
port=23466


# config=upernet_swin_base_patch4_window7_128x128_less_aug
config=upernet_swin_base_patch4_window7_512x512_less_aug

if [ $1 = "train" ]; then
    CUDA_VISIBLE_DEVICES=$GPU PORT=${port} ./tools/dist_train.sh configs/swin/${config}.py 4 --work-dir cache/${config} 
elif [ $1 = "test" ]; then
    CUDA_VISIBLE_DEVICES=$GPU ./tools/dist_test.sh configs/swin/${config}.py ./cache/${config}/best_mDice_epoch_20.pth 1 --format-only --work-dir cache/${config} --eval-options "imgfile_prefix=./test_results/${config}"
fi

