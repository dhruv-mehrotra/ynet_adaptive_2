seeds=(1 2 3) # Fine-tune the model on different seeds
batch_size=1

# seeds=(1 2 3) # Fine-tune the model on different seeds
# batch_size=8

num_epochs=50
foldername=sdd_ynet/ # See README.md how to setup this directory
out_csv_dir=csv # /path/to/csv where the output results are written to
val_ratio=0.5 # Split train dataset into a train and val split in case the domains are the same
dataset=dataset_filter/dataset_ped_biker/gap/ # Position the dataset in /path/to/sdd_ynet/

list_num_batches=(1 2 3 4 5) # Fine-tune the model with a given number of batches

train_net=modulator # Train either all parameters, only the encoder or the modulator: (all encoder modulator)


val_files='3.25_3.75.pkl' # Position the dataset files in /path/to/sdd_ynet/{dataset}
train_files='3.25_3.75.pkl' # Position the dataset files in /path/to/sdd_ynet/{dataset}

# ckpt=ckpts/Seed_1_Train__0.25_0.75__1.25_1.75__2.25_2.75__Val__0.25_0.75__1.25_1.75__2.25_2.75__Val_Ratio_0.3_dataset_ped_biker_gap_weights.pt # Pre-trained model
# ckpt=ckpts/Seed_1_Train__0.25_0.75__1.25_1.75__2.25_2.75__Val__0.25_0.75__1.25_1.75__2.25_2.75__Val_Ratio_0.3_dataset_ped_biker_gap_modular_weights.pt # Pre-trained model # LARGE LR

ckpt=ckpts/Seed_1_Train__0.25_0.75__1.25_1.75__2.25_2.75__Val__0.25_0.75__1.25_1.75__2.25_2.75__Val_Ratio_0.3_dataset_ped_biker_gap_weights.pt # Pre-trained model


for seed in ${seeds[@]}
do
    for num in ${list_num_batches[@]}
    do
        python train_SDD.py --fine_tune --seed $seed --batch_size $batch_size --foldername $foldername --val_ratio $val_ratio --dataset $dataset --val_files $val_files --out_csv_dir $out_csv_dir --num_epochs $num_epochs --train_files $train_files --num_train_batches $num --train_net $train_net --ckpt $ckpt --learning_rate 0.01
    done
done