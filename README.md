# Bangladeshi Taka Note Detection Using YOLOv8

This project folder contains all the necessary resources and trained models for the Bangladeshi Taka Note Detection project.

## Folder Structure
- dataset/: Contains a text file with the link to the actual dataset.
- 
otebook/: Contains the Jupyter notebook used for training and inference (BD_Taka_Detection_YOLOv8.ipynb).
- inference_results/: Contains sample inference images showing bounding boxes on test images, including custom test images and bonus task results (notes + coins).
- models/: Contains the final trained YOLOv8 weights:
  - 	aka_detection_best.pt: Model trained only on Taka notes.
  - combined_detection_best.pt: Model trained on both Taka notes and coins (Bonus Task).
- 	raining_eval_logs/: Contains the YOLOv8 uns directory, which encompasses all training logs, evaluation scripts, graphs, and metrics for both modules.
