python main.py --mode 'train' --device 'cuda' --datafile './sant/inputs/all_data.csv' --epochs 10 --learning_rate 1e-3 --batch_size 256 --learning_rate_decay 0.995 --regularization 1e-5 --random_seed 56 --hidden_dim 1500 --bottleneck 0 --resnet_trick 1 --bottleneck_dim -1 --num_hidden_layers 3 --checkpoint './sant/model_1.pth' --use_existing_checkpoint 1 --print_every 1 --normalize 'rankGauss' --noise 'permute' --noise_param 0.15 --out_file 'sant/model_1_activations.csv'