import argparse

defaults = {
    "dataset": 'CIFAR10',
    "decoder_type": 'Gaussian',
    "model_sigma": False,
    "epochs": 100,
    "batch_size": 100,
    "learning_rate": 5e-4,
    "latent_dim": 10,
    "print_every": 1,
    "resume_path": None,
    "resume_epoch": 0
}

def check_args(args):
    """Check commandline argument validity."""
    assert args.dataset=='MNIST' or args.dataset=='CIFAR10', "Dataset must be either 'MNIST' for 'CIFAR10'"

    assert args.decoder_type=='Gaussian' or args.decoder_type=='Bernoulli', "Decoder type must be either 'Gaussian' for 'Bernoulli"

    assert not (args.dataset=='CIFAR10' and args.decoder_type=='Bernoulli'), "Bernoulli decoder only supports MNIST"
    
    assert args.epochs >= 1, "Number of epochs must be a positive integer"

    assert args.batch_size >= 1, "Size of batch must be a positive integer"
    
    assert args.learning_rate > 0, "Learning rate must be positive"
    
    assert args.latent_dim >= 1, "Latent dimension must be a positive integer"
    
    assert args.print_every >= 1, "Print_every must be a positive integer"
    
    assert args.resume_epoch >= 0, "Resume_epoch must be a non-negative integer"
    
    return args

def get_args():
    """Parse arguments from commandline."""
    parser = argparse.ArgumentParser(
        description="Pytorch Implementation of Variational Autoencoder (VAE)")
    
    parser.add_argument("-d", "--dataset",
        type=str, default=defaults['dataset'], help="'MNIST' or 'CIFAR10'")

    parser.add_argument("-t", "--decoder_type",
        type=str, default=defaults['decoder_type'], help="'Gaussian' or 'Bernoulli'")

    parser.add_argument("-s", "--model_sigma",
        type=bool, default=defaults['model_sigma'], help="In case of Gaussian decoder, whether to model the standard deviation")
    
    parser.add_argument("-e", "--epochs",
        type=int, default=defaults['epochs'], help="Number of epochs to train")

    parser.add_argument("-b", "--batch_size",
        type=int, default=defaults['batch_size'], help="Size of batch at training/testing")
    
    parser.add_argument("-lr", "--learning_rate",
        type=float, default=defaults['learning_rate'], help="Learning rate for adam optimizer")
    
    parser.add_argument("-z", "--latent_dim",
        type=int, default=defaults['latent_dim'], help="Dimension of latent variable z")
    
    parser.add_argument("-p", "--print_every",
        type=int, default=defaults['print_every'], help="How often to print loss progress")

    parser.add_argument("-rp", "--resume_path",
        type=str, default=defaults['resume_path'], help="If we wish to resume training, provide the saved model path here")
    
    parser.add_argument("-re", "--resume_epoch",
        type=int, default=defaults['resume_epoch'], help="In case of resuming, the number of epochs already done")
    
    return check_args(parser.parse_args())