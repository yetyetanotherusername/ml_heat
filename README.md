# Machine Learning assisted heat detection

## Setup

Create virtual environment
```
python3 -m venv .env
```
Activate virtual environment
```
source .env/bin/activate
```
Install dependencies
```
pip install -Ur requirements.txt
```
or if you only need minimal dependencies for preprocessing
```
pip install -Ur requirements_preprocessing.txt
```
Run setup.py in develop mode
```
python setup.py develop
```

## Run Tests
```
pytest tests/
```

## Download data

This step is only possible with developer access to smaxtec google cloud

Establish port forwarding into the cluster
```
kubectl get pods
kubectl port-forward {podname} 8787:8787
```
Download data
```
python ml_heat/data_loading/get_data.py
```

## Prepare data

This step uses the downloaded rawdata, generates a few features and puts the data into a pandas dataframe ready for use

Perform preprocessing & data cleaning
```
python ml_heat/preprocessing/transform_data.py
```