pip install -r requirements.txt

gdown "https://drive.google.com/uc?export=download&id=1F6HAz15xSWQ6oyRgOPy_JP5qgSJvyt7f"
unzip output_videos.zip
rm output_videos.zip

gdown "https://drive.google.com/uc?export=download&id=1czq9F8TX8bUkq086IMDL-6H2BUe-GUbn"
unzip videos.zip
rm videos.zip

echo "Running lane detection..."
python lane.py

echo "Running 3D Bounding Box pipeline..."
python run.py

echo "Process completed successfully!"
