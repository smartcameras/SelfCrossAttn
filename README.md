# Cross- and self-attention for multi-modal emotion recognition

Code of the IEEE ICASSP 2022 paper "Is cross-attention preferable to self-attention for multi-modal emotion recognition?" <link to paper> (edited)


To download the pre-processed IEMOCAP dataset, use the link given in https://github.com/david-yoon/attentive-modality-hopping-for-SER
Once you have it downloaded, replace the 'data_path' in 'multi_run.sh' with your folder path.

Use the bash file 'multi_run.sh' to run the 5 fold cross validation with 10 runs on each fold. Remember to do 'chmod +x ./multi_run.sh' before executing the bash file.
