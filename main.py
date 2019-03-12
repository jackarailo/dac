import os
import argparse

import numpy as np
import torch

import model
import data
import train

parser = argparse.ArgumentParser(
            description=r"""Denoising MLP autoencoder""")
parser.add_argument('--mode', type=str, default='train',
                    help='mode to run main (train, generate)')
parser.add_argument('--device', type=str, default='cuda',
                    help='device to use (cpu, cuda)')
parser.add_argument('--datafile', type=str, 
                    default='/input/all_data.csv',
                    help='file with train and test data')
parser.add_argument('--epochs', type=int, default=100,
                    help=r"Number of epochs to train the model")
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help=r"Learning rate")
parser.add_argument('--learning_rate_decay', type=float, default=0.995,
                    help=r"Learning rate")
parser.add_argument('--regularization', type=float, default=0.0,
                    help=r"L2 Weights regularization")
parser.add_argument('--random_seed', type=int, default=-1,
                    help=r"Random seed for initialization")
parser.add_argument('--batch_size', type=int, default=64,
                    help=r"Batch size during training")
parser.add_argument('--hidden_dim', type=int, default=1500,
                    help=r"Dimension of hidden layers")
parser.add_argument('--bottleneck', type=int, default=0,
                    help=r"Bool if bottleneck is required")
parser.add_argument('--resnet_trick', type=int, default=1,
                    help=r"Apply resnet trick where possible")
parser.add_argument('--bottleneck_dim', type=int, default=-1,
                    help=r"Number of hidden layers")
parser.add_argument('--num_hidden_layers', type=int, default=3,
                    help=r"Number of hidden layers")
parser.add_argument('--checkpoint', type=str, default='./model_state_dict.pt',
                    help='state_dict checkpoint to use')
parser.add_argument('--use_existing_checkpoint', type=int, default=1,
                    help='use existing checkpoint if it exists')
parser.add_argument('--print_every', type=int, default=100,
                    help='Print to log file every n epochs')
parser.add_argument('--normalize', type=str, default='std',
                    help='normalizing scheme [std, minmax, rankGauss]')
parser.add_argument('--noise', type=str, default='permute',
                    help='normalizing scheme [permute, gauss]')
parser.add_argument('--noise_param', type=float, default='0.1',
                    help='permute prob for permute and gauss std for gauss noise')
parser.add_argument('--out_file', type=str, default='./data_activations.csv',
                    help='file to save activation data')

args = parser.parse_args()
# Globals
DATA_FILE = args.datafile
MODE = args.mode
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
PRINT_EVERY = args.print_every
LR = args.learning_rate
LR_DECAY = args.learning_rate_decay
REG = args.regularization
BOTTLENECK = args.bottleneck
RESNET_TRICK = args.resnet_trick
NBOT = args.bottleneck_dim
NHID = args.hidden_dim
NHLAYERS = args.num_hidden_layers
DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")
RANDOM_SEED = args.random_seed if args.random_seed != -1 else None
CHECKPOINT_FILE = args.checkpoint
USE_EXISTING_CHECKPOINT = False if args.use_existing_checkpoint == 0 else True
NORMALIZE = args.normalize
NOISE = args.noise if MODE == 'train' else None
NOISE_PARAM = args.noise_param if MODE == 'train' else None
OUT_FILE = args.out_file

def main():
    # Load the CSV dataframe
    print("------Loading Dataframe---------")
    data_df = data.load(DATA_FILE)
    print("------DataFrame Loading DONE--------")
    # Convert to numpy and normalize
    print("------Normalizing Data---------")
    train_data = data.normalize(data_df, NORMALIZE)
    print("------Normalizing Data DONE---------")
    # Create Pytorch dataloader
    dataloader = data.create_dataloader(train_data, BATCH_SIZE, DEVICE, 
                                        NOISE, NOISE_PARAM)
    print("------Created Dataloader---------")
    N, ninp = train_data.shape
    # CREATE MODEL
    if BOTTLENECK:
        net = model.DACBottle(ninp, NHID, NBOT, NHLAYERS, BOTTLENECK, 
                              RESNET_TRICK, RANDOM_SEED).to(DEVICE)
    else:
        net = model.DAC(ninp, NHID, NHLAYERS, RESNET_TRICK, RANDOM_SEED).to(DEVICE)
    print("------Loaded Model---------")
    N, ninp = train_data.shape
    if ((USE_EXISTING_CHECKPOINT or MODE == 'generate') 
        and os.path.isfile(CHECKPOINT_FILE)):
        print("----------Loading CHECKPOINT-----------------")
        checkpoint = torch.load(CHECKPOINT_FILE)
        net.load_state_dict(checkpoint['model_state_dict'])
        print("----------Loading CHECKPOINT DONE-----------------")
    if MODE == 'train':
        # GET NORM DATA WITH NOISE AND GENERATE PREDICTIONS
        print("-----------Starting training--------------")
        trainer = train.Trainer(net, LR, LR_DECAY, REG)
        best_loss = np.inf
        for i in range(EPOCHS):
            for bx, by in dataloader:
                bx = bx.to(DEVICE)
                by = by.to(DEVICE)
                loss = trainer.step(bx, by)
            if i % PRINT_EVERY == 0:
                print(f"Epoch: {i}\t Training Loss: {loss}")
            if loss < best_loss:
                best_loss = loss
                torch.save({'model_state_dict': net.state_dict()},
                            CHECKPOINT_FILE)
        print("-----------Training DONE--------------")
    elif MODE == 'generate':
        # GET CLEAN NORM DATA AND GENERATE FEATURES FROM ACTIVATIONS
        print("----------Generating FEATURES-----------------")
        model.eval()
        with torch.no_grad():
            all_data = []
            for bx, by in dataloader:
                x = bx.to(DEVICE)
                out = net.generate(x)
                if len(all_data) == 0:
                    all_data = out
                else:
                    all_data = np.vstack((all_data, out))
        np.savetxt(OUT_FILE, all_data, delimiter=",")
        print("----------FEATURES generated and saved to file------------")

def test():
    data_df = data.load("./sant/inputs/all_data.csv")
    print(data_df.columns)

if __name__ == "__main__":
    main()
