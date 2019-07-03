## ml-heat

# create virtual environment
python3 -m venv .env

# activate virtual environment
source .env/bin/activate

# install dependencies
pip install -Ur requirements.txt

# download data
python ml_heat/data_loading/get_data.py

# perform preprocessing
python ml_heat/preprocessing/transform_data.py