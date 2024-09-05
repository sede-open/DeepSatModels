# > bash remove_temp_files.sh

find "$PWD" -type f -name "*muhammed*" -delete
find "$PWD" -type f -name "*.amltmp" -delete
find "$PWD" -type f -name "*.amlignore" -delete
find "$PWD" -type f -name "*.pyc" -delete

find "$PWD" -type d -name "__pycache__" -delete
find "$PWD" -type d -empty -delete

ls -1R
