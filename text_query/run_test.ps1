# Run this from text_query/ to download 2 Epic Kitchen clips and test the pipeline
# Activate conda first: conda activate ren_venv

# Step 1: Download 2 real Epic Kitchen videos (participant P01, first 2 min each)
python download_epic_kitchen.py --real --participant P01 --limit 2 --trim-start 30 --trim-duration 120 --output ..\epic_kitchen_data\

# Step 2: Index the first video
python prepare_index.py "..\epic_kitchen_data\P01_01.mp4" --output ..\epic_kitchen_indexes\P01_01\ --sample-rate 2

# Step 3: Query
python query_indexed.py "knife" --index ..\epic_kitchen_indexes\P01_01\ --video "..\epic_kitchen_data\P01_01.mp4" --output ..\epic_results\knife_test\ --threshold 0.15

# Step 4: View results (Windows)
Invoke-Item "..\epic_results\knife_test\last_occurrence.mp4"
Get-Content "..\epic_results\knife_test\result.json"
