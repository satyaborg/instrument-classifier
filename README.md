# Instrument Classification

Deep learning based instrument classification on OpenMIC-2018 dataset.

## Getting Started

> Note: Fully tested on WSL2 with Python 3.11.0 on NVIDIA GeForce RTX 2080 (CUDA 12.2).

Clone this repository and change directory (cd) into it.

Make sure to create the following data folders under disco by running:

```
mkdir -p disco/data/raw disco/data/processed disco/data/augmentations

```

Full directory structure [here](#misc).

### 0. Download Dataset

Download the OpenMIC-2018 dataset from [here](https://zenodo.org/record/1432913#.Yb6Q9pNKhTY) and place `openmic-2018-v1.0.0.tgz` and untar it under `disco/data/raw/` as follows:

```
cd disco/data/raw
tar -xzvf openmic-2018-v1.0.0.tgz

```

### 1. Environment Setup

To setup your environment and project, run the following commands:

```
sudo apt-get install python3.11
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

This should set up all dependencies to run the project.

### 2. Preprocessing

To run the preprocessing pipeline on the dataset, run the following command:

```
python3 -m disco.scripts.preprocess
```

> Note: This will take some time but it is a one-off process that will make training faster.

The preprocessing involves:

- Resampling all audio files to 16kHz and mono channel
- Optionally augmenting the dataset with, pitch and time shifts, for example
- Feature extraction from pretrained VGGish model
- Exporting embeddings to TFRecords for native TF/keras support

Elapsed times (per tested system):

- 61.48 mins w/ augmentations
- 16.246 mins w/o augmentations

> Note: Transfer learning was also considered by allowing VGGish to be the base model (with frozen and trainable for the early and later ones respectively). Due to lack of resources, time constraints and issues with tensor dimension issues when intializing `hub.KerasLayer`, hence was not pursued further.

### 3. Configuration

To configure the hyperparameters, model spec etc. please edit the `disco/config.py` file. Once done, you can proceed with training the model.

> Note: This can be improved further by setting up grid search (or Bayesian optimisation) for hyperparameter tuning using [Keras Tuner](https://keras.io/api/keras_tuner/).

### 3. Training

To train the model, execute the following:

```

python3 -m disco.scripts.train

```

> Note: All model artifacts will be saved under the `results/` folder.

### 4. Analysis

To analyse the different runs, execute the following:

```
python3 -m disco.scripts.analyse
```

This will generate a `results.csv` file under the `results/` folder, which can be used to compare the different run, along with plots for loss and metrics.

## OpenMIC-2018 Dataset

- 20 instrument classes
  - accordion, banjo, bass, bassoon, cello, clarinet, erhu, flute, guitar, horn, mallet, oboe, organ, piano, saxophone, trombone, trumpet, ukulele, violin, and voice
- 20,000 unique tracks
  - 18,000 training tracks
  - 2,000 test tracks
- Class frequency

```
Instrument         Frequency
violin               1188
piano                1161
saxophone            1135
guitar               1124
flute                1117
synthesizer          1090
drums                1073
mallet_percussion    1046
voice                1043
organ                1038
cymbals              1016
cello                1011
trumpet              1001
bass                  948
banjo                 947
ukulele               939
trombone              870
accordion             843
mandolin              832
clarinet              578
```

### Performance

Based on the simple approach of using a pretrained model as a feature extractor for a dense classifier (w/ and w/o augmentations), we're able to achieve a f1-score of `0.6` on the test set. Precision and recall are `0.76` and `0.42` respectively.

![metrics](/disco/results/metrics.png)

The hyperparameters that worked the best:

- `batch_size`: `64`
- `epochs`: `69` (with early stopping)
- `learning_rate`: `1e-04`
- `dropout_rate`: `0.5`
- `l2_reg`: `1e-02`

<table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th>runid</th>      <th>model_name</th>      <th>seed</th>      <th>epochs</th>      <th>batch_size</th>      <th>learning_rate</th>      <th>dropout</th>      <th>l2_regularization</th>      <th>test_loss</th>      <th>test_precision</th>      <th>test_recall</th>      <th>test_f1_score</th>    </tr>  </thead>  <tbody>    <tr>      <td>1705569373</td>      <td>vggish</td>      <td>42</td>      <td>69</td>      <td>64</td>      <td>0.0001</td>      <td>0.5</td>      <td>0.010000</td>      <td>1.619573</td>      <td>0.762968</td>      <td>0.416519</td>      <td>0.600590</td>    </tr>    <tr>      <td>1705571710</td>      <td>vggish</td>      <td>42</td>      <td>96</td>      <td>128</td>      <td>0.0001</td>      <td>0.5</td>      <td>0.010000</td>      <td>1.704298</td>      <td>0.775157</td>      <td>0.413569</td>      <td>0.599213</td>    </tr>    <tr>      <td>1705571134</td>      <td>vggish</td>      <td>42</td>      <td>71</td>      <td>128</td>      <td>0.0001</td>      <td>0.5</td>      <td>0.010000</td>      <td>1.613300</td>      <td>0.768100</td>      <td>0.400590</td>      <td>0.599017</td>    </tr>    <tr>      <td>1705568992</td>      <td>vggish</td>      <td>42</td>      <td>46</td>      <td>64</td>      <td>0.0001</td>      <td>0.2</td>      <td>0.010000</td>      <td>1.626133</td>      <td>0.765208</td>      <td>0.405703</td>      <td>0.591347</td>    </tr>    <tr>      <td>1705568634</td>      <td>vggish</td>      <td>42</td>      <td>26</td>      <td>64</td>      <td>0.0001</td>      <td>0.2</td>      <td>0.000010</td>      <td>1.413879</td>      <td>0.743647</td>      <td>0.454671</td>      <td>0.590167</td>    </tr>    <tr>      <td>1705568392</td>      <td>vggish</td>      <td>42</td>      <td>24</td>      <td>64</td>      <td>0.0001</td>      <td>0.2</td>      <td>0.000001</td>      <td>1.406683</td>      <td>0.737410</td>      <td>0.443461</td>      <td>0.588397</td>    </tr>    <tr>      <td>1705567634</td>      <td>vggish</td>      <td>42</td>      <td>13</td>      <td>64</td>      <td>0.0010</td>      <td>0.2</td>      <td>0.000000</td>      <td>1.432302</td>      <td>0.730506</td>      <td>0.440315</td>      <td>0.573845</td>    </tr>    <tr>      <td>1705565483</td>      <td>vggish</td>      <td>42</td>      <td>3</td>      <td>64</td>      <td>0.0001</td>      <td>0.2</td>      <td>0.000000</td>      <td>6.433951</td>      <td>0.788425</td>      <td>0.326844</td>      <td>0.570698</td>    </tr>    <tr>      <td>1705569971</td>      <td>vggish</td>      <td>42</td>      <td>83</td>      <td>64</td>      <td>0.0010</td>      <td>0.5</td>      <td>0.010000</td>      <td>1.723113</td>      <td>0.770225</td>      <td>0.363225</td>      <td>0.569518</td>    </tr>    <tr>      <td>1705565383</td>      <td>vggish</td>      <td>42</td>      <td>3</td>      <td>64</td>      <td>0.0001</td>      <td>0.2</td>      <td>0.000000</td>      <td>6.449058</td>      <td>0.788517</td>      <td>0.324090</td>      <td>0.567355</td>    </tr>  </tbody></table>

Current SOTA performance on OpenMIC-2018:

> Note: The authors report Mean average precision (mAP) instead of f1-score.

- DyMN-L\*\*: `0.855`
- EAsT-KD + PaSST: `.852`
- EAsT-Final + PaSST: `.847`
- PaSST: `.843`

\*\*uses extra data

## Future Considerations

- [YAMnet](https://www.kaggle.com/models/google/yamnet/frameworks/tensorFlow2/variations/yamnet/versions/1) was considered as another viable alternative but decided against due to higher memory requirements (e.g. the 1-D embedding size is 20480) as compared to VGGish, which is comparitively smaller, and also pretrained on AudioSet. It can be revisited as part of an ensemble model, where low confidence classes can be complemented by more semantically and acoustically rich embeddings down the track

- [Musicnn](https://github.com/jordipons/musicnn/tree/master) would also be an excellent candidate, given there's a significant overlap between the classes in OpenMIC-2018 and the ones in the [MagnaTagATune](https://github.com/keunwoochoi/magnatagatune-list) (MTT) dataset that was used for pretraining, hence exploiting the transfer learning potential of the model

- Additional datasets can be leveraged, as exemplified by the current SOTA approach on OpenMIC-2018, [DyMN-L](https://paperswithcode.com/paper/dynamic-convolutional-neural-networks-as)

- More data augmentation techniques can also be explored, such as [SpecAugment](https://arxiv.org/abs/1904.08779) and [Mixup](https://arxiv.org/abs/1710.09412) for Mel-spectrograms

## Troubleshooting

- Note that `librosa` is not compatible with Python `3.12`

## References

- Humphrey, Eric J., Durand, Simon, and McFee, Brian. "OpenMIC-2018: An Open Dataset for Multiple Instrument Recognition." in Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR), 2018.
- https://paperswithcode.com/sota/instrument-recognition-on-openmic-2018

## Misc

- Directory structure

```
.
├── disco/
│   ├── data/
│   │   ├── augmented/
│   │   │   ├── vggish_features
│   │   │   └── ..
│   │   ├── processed/
│   │   │   ├── vggish_features
│   │   │   └── ..
│   │   └── raw/
│   │       ├── openmic-2018
│   │       └── openmic-2018-v1.0.0.tgz
│   ├── models/
│   │   ├── dense.py
│   │   └── ..
│   ├── results/
│   │   ├── results.csv
│   │   └── ..
│   ├── scripts/
│   │   ├── analyse.py
│   │   ├── augment.py
│   │   ├── preprocess.py
│   │   └── train.py
│   ├── utils/
│   │   ├── dataloader.py
│   │   ├── helpers.py
│   │   └── trainer.py
│   └── config
├── .gitignore
├── venv
├── README.md
└── requirements.txt
```
