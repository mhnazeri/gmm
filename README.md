PyTorch implementation of GMM-TF.

## Dependencies
The main dependencies are PyTorch and nuScenes devkit. You can install other dependencies by running:
```bash
pip install -r requirements
```
## Preprocess the dataset
First we need to extract meta data from nuScenes dataset by running:
```bash
python data/nuscene_data_loader.py NUSCENES_DIR
```
where `$NUSCENES_DIR` is the absolute address to nuScenes directory. It will create a folder called `meta_data` which includes a json file for each scene containing the meta data. You can change the name of the folder by passing `--source` argument.

Then we need to create samples from meta data:
```bash
# create train samples
python data/data_helpers.py NUSCENES_DIR
# create validation data
python data/data_helpers.py NUSCENES_DIR --source meta_data_val --save_dir val_data
```
It will create a directory containing samples in `.pt` extension. Another argument that you can pass is the `--arch` argument to specify the architecture of context feature extractor. Currently, valid arguments are `overfeat` and `vgg`.

## Training the models
Train process consists of three parts. Train `CAE` and `CFEX` independently and train the whole model.
### Contractive Auto-encoder (CAE)
To train `CAE`, run following command:
```bash
python cae.py
```
You can change `CAE` parameters via `config.ini` file. (Note: remember after the training of `CAE`, set CAE's epochs in `config.ini` file to `0`.)
And finally, move the best checkpoint into cae directory (ex. assuming that checkpoint 100 is the best):
```bash
mkdir save/cae && cp save/cae_test/2/checkpoint-100.pt save/cae/best.pt
```

### Contextual Feature Extractor (CFEX)
Now is the time to train `CFEX`. You can also change `CFEX` parameters from `config.ini` file like `CAE`, and after the training, move the best checkpoint to `cfex` directory:
```bash
python cfex.py
# copy the best model to the cfex directory
mkdir save/cfex && cp save/cfex_train/checkpoint-100.pt save/cfex/best.pt
```
### GAN
At last, it is time to train the main model. You can change the model parameters from `config.ini`:
```bash
python train.py
```
