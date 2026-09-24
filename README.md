# Curvature-Transformer

Time series classification with Transformers (1D ViT / PatchTST) whose patches are either fixed-size or placed by curvature (the magnitude of the second derivative), with each variable-length patch zero-padded to the series length.

## Setup

```bash
git clone <this repository>
cd Curvature-Transformer

# The container user is UID/GID 1000 by default. If your host user differs, write yours to .env
# so the files created in the container (models/, outputs/) keep your ownership.
printf 'UID=%s\nGID=%s\n' "$(id -u)" "$(id -g)" > .env

# Build the image (CUDA 11.8 + Python 3.12 + dependencies via uv), start the container and enter it
docker compose up -d --build
docker compose exec app bash
```

## Data

The experiments use the [UCR Time Series Classification Archive (2018)](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).

1. Download `UCRArchive_2018.zip` from the archive page (the password is given in the archive's briefing document).
2. Extract it and place each dataset under `data/` as follows:

```
data/
├── ECG200/
│   ├── ECG200_TRAIN.tsv
│   └── ECG200_TEST.tsv
└── ...
```

## Usage

Inside the container:

```bash
bash scripts/train_test.sh ECG200
```

It trains for 100 epochs and then tests the saved weights. To change the patching, the model or the hyperparameters, edit the options in `scripts/train_test.sh`. The accuracy is printed and appended to `outputs/txt/out.txt`, and the weights are saved under `models/`.
