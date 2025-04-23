python -m venv env 
source env/bin/activate
pip install -r requirements.txt

echo "Running lane.py..."
python lane.py

echo "Running run.py..."
python run.py