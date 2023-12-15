# Document Segmentation

## Usage

Run the [scripts/evaluate.py](scripts/evaluate.py) script.
It downloads the necessary data from the HUC server into the local temporary directory.

Set your HUC credentials in the `HUC_USER` and `HUC_PASSWORD` environment variables, and run the script.
For instance:

```console
HUC_USER=... HUC_PASSWORD=... python scripts/evaluate.py
```

## Development Instructions

This project uses Python 3.11 and Poetry.

### Install Poetry (see [instructions](https://python-poetry.org/docs/master/#installation))

```console
curl -sSL https://install.python-poetry.org | python3 -
```

### Clone the repository

```console
git clone git@github.com:LAHTeR/document_segmentation.git
```

### Install dependencies

```console
poetry install --with=dev
```

### Set up pre-commit hooks

Install pre-commit:

```console
pre-commit install
```
