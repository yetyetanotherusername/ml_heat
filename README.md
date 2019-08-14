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

## Download & prepare data for machine learning

Download rawdata (only possible with developer access to smaxtec google cloud)
```
python ml_heat/data_loading/get_data.py
```
Perform preprocessing & data cleaning
```
python ml_heat/preprocessing/transform_data.py
```
