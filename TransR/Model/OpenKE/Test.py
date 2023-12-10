import sys
import os
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransR
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# Hyperparameters and key parameters

# Data Loading Parameters
nbatches = 1000            # Number of batches for training data
threads = 8                # Number of threads for data loading
neg_ent = 25               # Number of negative entities to sample

# Model Hyperparameters
dim_e = [100, 100, 50, 100]               # Dimensionality of entity embeddings (TransR)
dim_r = [100, 100, 50, 100]                # Dimensionality of relation embeddings (TransR)

# Creating an instance of TestDataLoader to load the testing data
test_dataloader = TestDataLoader(
	in_path = "./benchmarks/lastfm/",       # Path to the input data for testing
	sampling_mode = 'link',      # The sampling mode for testing (e.g., 'link', 'tail prediction')
    type_constrain = False)      # Flag for type constraint (True or False)

# test the model
checkpoint_path = '/content/drive/My Drive/TransR/Model/OpenKE/checkpoint/'
result_path = '/content/drive/My Drive/TransR/Result/'

# Ensure the result directory exists
if not os.path.exists(result_path):
    os.makedirs(result_path)

# List of your checkpoint files
checkpoints = ['lastfm_v1.ckpt', 'lastfm_v2.ckpt', 'lastfm_v3.ckpt', 'lastfm_v4.ckpt']

# Iterate over the checkpoint files
for i, checkpoint_file in enumerate(checkpoints):
    transr = TransR(
    ent_tot = test_dataloader.get_ent_tot(),
    rel_tot = test_dataloader.get_rel_tot(),
    dim_e = dim_e[i],                               # Dimensionality of the entity embeddings
    dim_r = dim_r[i],                               # Dimensionality of the relation embeddings
    p_norm = 1,                                # Norm used for distance calculation in loss function (1 for L1 norm)
    norm_flag = True,
    rand_init = False)                         # Flag to control random initialization of embeddings
    
    # Load the checkpoint for the current model
    model_path = os.path.join(checkpoint_path, checkpoint_file)
    transr.load_checkpoint(model_path)
    
    # Create an instance of the Tester
    tester = Tester(model=transr, data_loader=test_dataloader, use_gpu=True)
    
    # Redirect standard output to a file
    result_file = os.path.join(result_path, checkpoint_file.replace('.ckpt', '_test_results.txt'))
    with open(result_file, 'w+') as f:
        sys.stdout = f
        # Call the function that runs the test and prints the results
        tester.run_link_prediction(type_constrain=False)
        # Reset standard output
        sys.stdout = sys.__stdout__
    
    print(f'Results for {checkpoint_file} have been saved to {result_file}.')
