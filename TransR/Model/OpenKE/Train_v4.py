import sys
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE, TransR
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# Hyperparameters and key parameters

# Data Loading Parameters
nbatches = 400          # Number of batches for training data
threads = 8                # Number of threads for data loading
neg_ent = 25               # Number of negative entities to sample

# Model Hyperparameters
dim = 100                # Dimensionality for embeddings (used in both TransE and TransR)
dim_e = 100                # Dimensionality of entity embeddings (TransR)
dim_r = 100                # Dimensionality of relation embeddings (TransR)

# Training Parameters
train_times_transe = 1       # Number of training epochs for TransE
train_times_transr = 400     # Number of training epochs for TransR
alpha_transe = 0.5         # Learning rate for TransE
alpha_transr = 1.0         # Learning rate for TransR

# Loss Function Margins
margin_transe = 5.0        # Margin for TransE model
margin_transr = 3.0        # Margin for TransR model



# Creating an instance of TrainDataLoader to load the training data
train_dataloader = TrainDataLoader(
    in_path = "./benchmarks/lastfm/",  # Path to the input data within the current directory
    nbatches = nbatches,       # The number of batches in which the data is to be divided
    threads = threads,           # Number of threads to use for loading data
    sampling_mode = "normal",  # The sampling mode for training (e.g., 'normal', 'bernoulli')
    bern_flag = 1,         # Flag to indicate if Bernoulli sampling is used (1 for true, 0 for false)
    filter_flag = 1,       # Flag to indicate if filtering is applied (1 for true, 0 for false)
    neg_ent = neg_ent,          # Number of negative entities to sample
    neg_rel = 0)           # Number of negative relations to sample

# Creating an instance of TestDataLoader to load the testing data
test_dataloader = TestDataLoader(
	in_path = "./benchmarks/lastfm/",       # Path to the input data for testing
	sampling_mode = 'link',      # The sampling mode for testing (e.g., 'link', 'tail prediction')
    type_constrain = False)      # Flag for type constraint (True or False)

# Definition of the TransE model
transe = TransE(
    ent_tot = train_dataloader.get_ent_tot(),  # Total number of entities in the dataset
    rel_tot = train_dataloader.get_rel_tot(),  # Total number of relations in the dataset
    dim = dim,                                 # Dimensionality of the entity and relation embeddings
    p_norm = 1,                                # Norm used for distance calculation in loss function (1 for L1 norm)
    norm_flag = True)                          # Flag to normalize the embeddings

# Wrapping the TransE model with NegativeSampling and specifying the loss function
model_e = NegativeSampling(
    model = transe,                            # The TransE model defined above
    loss = MarginLoss(margin = margin_transe),           # Margin-based loss with a margin of 5.0
    batch_size = train_dataloader.get_batch_size())  # Batch size for training

# Definition of the TransR model
transr = TransR(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim_e = dim_e,                               # Dimensionality of the entity embeddings
    dim_r = dim_r,                               # Dimensionality of the relation embeddings
    p_norm = 1,                                # Norm used for distance calculation in loss function (1 for L1 norm)
    norm_flag = True,
    rand_init = False)                         # Flag to control random initialization of embeddings

# Wrapping the TransR model with NegativeSampling and specifying the loss function
model_r = NegativeSampling(
    model = transr,                            # The TransR model defined above
    loss = MarginLoss(margin = margin_transr),           # Margin-based loss with a margin of 4.0
    batch_size = train_dataloader.get_batch_size())  # Batch size for training


# Pretraining the TransE model
trainer = Trainer(
    model = model_e,
    data_loader = train_dataloader,
    train_times = train_times_transe,                       # Number of epochs to train
    alpha = alpha_transe,                           # Learning rate
    use_gpu = True)
trainer.run()

# Getting the trained parameters from the TransE model
parameters = transe.get_parameters()

# Training the TransR model

# Initializing TransR with parameters from TransE
transr.set_parameters(parameters)
trainer = Trainer(
    model = model_r,
    data_loader = train_dataloader,
    train_times = train_times_transr,                     # Number of epochs to train
    alpha = alpha_transr,                           # Learning rate
    use_gpu = True)
trainer.run()

# Saving the trained TransR model
transr.save_checkpoint('./checkpoint/lastfm_v4.ckpt')

# test the model
transr.load_checkpoint('./checkpoint/lastfm_v4.ckpt')
tester = Tester(model = transr, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)
