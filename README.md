# Black Box Optical Character Recognition

## Summary

An image-to-seq tensorflow neural network designed to translate sheetmusic staffs into a vector of notation. The network has no hardcoded knowledge of the structure of sheetmusic. The training dataset is derived from the publically available [RISM data](https://opac.rism.info/index.php?id=10) but this network could be trained on any (image, glyph vector) dataset.

Check out [this presentation](https://docs.google.com/presentation/d/18cR_qtLp4KMEHGnGR7YbWSbYQRxGXgPCNgCywaA6rPI/edit?usp=sharing) for more information.

## Environment Set Up

The easiest way to get set up is with Anaconda

1. Download and install [Anaconda](https://www.anaconda.com/distribution/) for your system
2. Run the following commands from a conda enabled terminal to create a virtual enviroment and install the necessary packages

```terminal
    conda create --name tf-gpu
    conda activate tf-gpu
    conda install tensorflow-gpu keras-gpu
    conda install -c conda-forge opencv
    conda install jupyter
```

3. Clone this repository

## How To Train the Model

1. Download the full training data from [here](https://neural-network-omr-training.s3.amazonaws.com/Data.zip) and extract into a directory called `Data` within the repository directory
2. Open `OMR.ipynb` in jupyter notebook from the `tf-gpu` environment.
3. Allow the notebook to automatically run the first two setup cells.
4. Run the cell that starts the model training.

## How to Predict with the Model

There is a pretrained version of the model available to download to get started predicting

1. Download the pretrained model [here]([FIX FIX](https://neural-network-omr-inference.s3.amazonaws.com/Models.zip)) and extract into a folder called `Models` within the repository directory
2. Download a sample of the training data from [here]([FIX FIX](https://neural-network-omr-inference.s3.amazonaws.com/Data.zip)) and extract into a folder called `Data`
3. Open `OMR.ipynb` in jupyter notebook from the `tf-gpu` environment.
4. Allow the notebook to automatically run the first two setup cells.
5. Run the cells that perform the inference on the sample data.
