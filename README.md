# 7008 Project

This is a simple introduction about our group's project.

> Group members: Tong Shenyang, Wang Pengxin, Zhan Xiangcheng

Our chosen topic is **Recommendation Systems**. 

As per the requirements, we have completed implementations for the following algorithms:

1. Typical recommendation algorithms: Content-based filtering, Item-based collaborative  filtering, and User-based collaborative filtering
2. Deep learning-based algorithms: Neural Collaborative, graph neural network-based recommendation algorithms
3. TransR-Learning Entity and Relation Embeddings for Knowledge Graph Completion-AAAI2015

Next, we will provide instructions on how to run the algorithms for our project:

## Typical recommendation algorithms

In this section, our aim is to analyze the MovieLens-1M and Last.FM datasets using Content-based filtering, Item-based collaborative filtering, and User-based collaborative filtering respectively. The `Typical_for_ml-1m.ipynb` file encompasses all the processing and analysis procedures for the ml-1m dataset. Similarly, `Typical_hetrec2011-lastfm-2k.ipynb` contains all the processing and analysis procedures for the hetrec2011-lastfm-2k dataset.

Using the code is straightforward; simply open the file and continuously run it. This will generate the desired results. For instance, in `Typical_for_ml-1m.ipynb`, it will generate recommended movie names for a user with a specific user-id. In `Typical_hetrec2011-lastfm-2k.ipynb`, it will generate recommended artists for a user with a specific user-id.

## Neural Collaborative Filtering

The basic code is from

[hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering). 

It is the implementation for the paper:

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering.](http://dl.acm.org/citation.cfm?id=3052569) In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

Three collaborative filtering models: Generalized Matrix Factorization (GMF), Multi-Layer Perceptron (MLP), and Neural Matrix Factorization (NeuMF). 

For our project objectives, we add two `experiment.ipynb` files for data processing about `movielens` and `Last. FM` datasets. 


The following is the information about the author.

>Author: Dr. Xiangnan He (http://www.comp.nus.edu.sg/~xiangnan/)

### Environment Settings 
We try the codes on Windows with docker.
- Keras version:  '1.0.7'
- Theano version: '0.8.0'


### Docker Quickstart
Docker quickstart guide can be used for evaluating models quickly.


Build a keras-theano docker image 
```
docker build --no-cache=true -t ncf-keras-theano .
```

#### Example to run the codes with Docker.
Run the docker image with a volume (Run GMF):
```
docker run --volume=$(pwd):/home ncf-keras-theano python GMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
```

Run the docker image with a volume (Run MLP):
```
docker run --volume=$(pwd):/home ncf-keras-theano python MLP.py --dataset ml-1m --epochs 20 --batch_size 256 --layers [64,32,16,8] --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
```

Run the docker image with a volume (Run NeuMF -without pre-training): 
```
docker run --volume=$(pwd):/home ncf-keras-theano python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --reg_mf 0 --reg_layers [0,0,0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
```

Run the docker image with a volume (Run NeuMF -with pre-training):
```
docker run --volume=$(pwd):/home ncf-keras-theano python NeuMF.py --dataset ml-1m --epochs 20 --batch_size 256 --num_factors 8 --layers [64,32,16,8] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1 --mf_pretrain Pretrain/ml-1m_GMF_8_1501651698.h5 --mlp_pretrain Pretrain/ml-1m_MLP_[64,32,16,8]_1501652038.h5
```

* **Note**: `--volume=$(pwd)` may have trouble when using in Windows. We recommend to use absolute path instead. The command can be written as 

```
docker run -v  ** Your absolute path**:/home:rw ncf-keras-theano python GMF.py --dataset Lastfm --epochs 20 --batch_size 256 --num_factors 8 --regs [0,0] --num_neg 4 --lr 0.001 --learner adam --verbose 1 --out 1
```


### Dataset
We provide two raw datasets: MovieLens 1 Million and Last. FM and relevant `.ipynb` file for data processing. Except for two datasets mentioned in project objectives, Pinterest (pinterest-20) also in the Data directory and ready for training and testing. 

In Data Directory you can find the processed datasets:
* MovieLens 1 Million (ml-1m) 
* Last. FM
* Pinterest (pinterest-20).

train.rating: 
- Train file.
- Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

test.rating:
- Test file (positive instances). 
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

test.negative
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.  
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...

# TransR Part

This repository contains the TransR model's implementation within the OpenKE framework for knowledge embedding. It's organized into directories for the model, datasets, and output results. 

## Personal Note

- In the whole project, I'm in charge of TranR part(including code, report, ppt) and PPT integration.
- I have invested considerable time in this project, especially in data preprocessing, path handling, and environment setup. I know the code looks like a mess, but this is already the best I can do. The majority of the development was done using Google Colab due to issues running the code locally.
- The OpenKE framework is cloned directly from its GitHub repository.

## Directory Structure

- `Result/`              # Contains the outcomes from model executions.
- `Model/`               # Houses the model code and associated utilities.
    - `OpenKE/`          # Core directory for the OpenKE framework.
        - `openke/`      # Core OpenKE module with foundational classes and methods.
	    - `make.sh`      # Shell script to compile necessary C++ operations.
            - `bash.ipynb`   # Jupyter notebook with shell commands for setup and execution.
        - `examples/`    # Example scripts to demonstrate OpenKE usage.
        - `checkpoint/`  # Stores model state checkpoints.
        - `benchmarks/`  # Includes datasets and scripts for benchmarks.
        - `Train_v*.py`  # Versioned Python scripts for model training.
- `Data/`                # Contains datasets for model training and testing.
    - `Raw/`             # Unprocessed data files.
        - `lastfm_raw/`  # Specific raw dataset for LastFM application.
    - `Processed/`       # Data that has been processed and is ready for use.
        - `lastfm/`      # Processed LastFM dataset.
    - `main.ipynb`       # Jupyter notebook for running data processing.
    - `data_preprocessing.py`  # Script for data preprocessing.
    - `data_preprocessing.ipynb`  # Jupyter notebook for data preprocessing.

## Challenges Faced

- C++ acceleration is used, which requires running `make.sh` in the `/openke` directory; however, attempts to run this locally often led to crashes.
- Manipulating the OpenKE package proved difficult, particularly with the Tester module, which prints but does not return results.

## Running the Code

1. Upload the whole directory on Google Colab Cloud Drive.
2. You may need to remove the `/release` directory before running the `make.sh` script. Compile the C++ components by executing `./make.sh` within the Model/OpenKE directory.
3. Data Preprocessing:
   - Navigate to the Data directory.
   - Run `python data_preprocessing.py` or execute `data_preprocessing.ipynb` in Jupyter.
4. Model Training and Testing:
   - Navigate to the Model/OpenKE directory.
   - To train the model, run the appropriate versioned training script, e.g., `python Train_v4.py`.
   - To test and see results, run `python Test.py` in the `/OpenKE` directory.

## Additional Notes

- If encountering any issues with the `make.sh` script, ensure that all C++ dependencies are installed and consider trying on a different system or Google Colab.
- The project is designed with modifiability in mind, but due to the complexity of OpenKE, deeper changes may require extensive understanding of the framework.

This project is a testament to the learning process and the challenges of working with complex frameworks and data. Thank you for your understanding and support.



## graph neural network-based recommendation algorithms

In this section, we chose the Light-GCN model to process the data and visualized metrics such as accuracy and regression of the results.

The usage involves entering the following code in Shell:

1. First, open a shell at `.\LightGCN-PyTorch\` location and execute:

``` bash
cd code
```

2. Adjust the desired parameters such as learning rate, number of model layers, and so on, then execute the code.

```bash
python main.py --decay=1e-4 --lr=0.001 --layer=4 --seed=2020 --dataset="ml-1m" --topks="[20]" --recdim=32
```

Below are the results of our model on the two datasets.

> all metrics is under 
>
> topks=20
>
> decay=1e-4
>
> learning rate=0.001
>
> seed=2020
>
> recdim=32

> - ml-1m stop at 50 epochs

|             | Recall | ndcg | precision |
| ----------- | - | ----------------- | ---- |
| **layer=1** | 0.0363        | 0.0375 | 0.02744 |
| **layer=2** | 0.0347                 | 0.0377 | 0.02870 |
| **layer=3** | 0.0403           | 0.0481 | 0.03934 |
| **layer=4** | 0.0443          | 0.0569 | 0.04863 |


> - ml-1m stop at 100 epochs


|             | Recall | ndcg | precision |
| ----------- | - | ----------------- | ---- |
| **layer=1** | 0.0356          | 0.0352 | 0.02457 |
| **layer=2** | 0.0348                  | 0.0351 | 0.02481 |
| **layer=3** | 0.0331             | 0.0343 | 0.02548 |
| **layer=4** | 0.0343            | 0.0379 | 0.03433 |

> - hetrec2011-lastfm-2k stop at 100 epochs


|             | Recall | ndcg | precision |
| ----------- | - | ----------------- | ---- |
| **layer=1** | 0.2268           | 0.1690 | 0.06281 |
| **layer=2** | 0.2524               | 0.1926 | 0.07053 |
| **layer=3** | 0.2605         | 0.2027 | 0.07312 |
| **layer=4** | 0.2656           | 0.2051   | 0.07449 |

> **Extend:**

* If you want to run lightGCN on your own dataset, you should go to `dataloader.py`, and implement a dataloader inherited from `BasicDataset`.  Then register it in `register.py`.

* If you want to run your own models on the datasets we offer, you should go to `model.py`, and implement a model inherited from `BasicModel`.  Then register it in `register.py`.

* If you want to run your own sampling methods on the datasets and models we offer, you should go to `Procedure.py`, and implement a function. Then modify the corresponding code in `main.py`


