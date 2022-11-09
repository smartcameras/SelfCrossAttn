# Is cross-attention preferable to self-attention for multi-modal emotion recognition?
<p align="center">
Vandana Rajan*<sup>1</sup> , Alessio Brutti<sup>2</sup> , Andrea Cavallaro<sup>1</sup> </br>
<sup>1</sup> Queen Mary University of London, London, United Kingdom</br>
<sup>2</sup> Fondazione Bruno Kessler, Trento, Italy</br>
*v.rajan@qmul.ac.uk</br>
</p>

# Abstract
Humans express their emotions via facial expressions, voice intonation and word choices. To infer the nature of the underlying emotion,
recognition models may use a single modality, such as vision, audio, and text, or a combination of modalities. Generally, models
that fuse complementary information from multiple modalities outperform their uni-modal counterparts. However, a successful model
that fuses modalities requires components that can effectively aggregate task-relevant information from each modality. As crossmodal
attention is seen as an effective mechanism for multi-modal fusion, in this paper we quantify the gain that such a mechanism
brings compared to the corresponding self-attention mechanism. To this end, we implement and compare a cross-attention and a selfattention
model. In addition to attention, each model uses convolutional layers for local feature extraction and recurrent layers for
global sequential modelling. We compare the models using different modality combinations for a 7-class emotion classification
task using the IEMOCAP dataset. Experimental results indicate that albeit both models improve upon the state-of-the-art in terms
of weighted and unweighted accuracy for tri- and bi-modal configurations, their performance is generally statistically comparable. The paper has been accepted in ICASSP 2022. 

# Data
To download the pre-processed IEMOCAP dataset, use the link given in https://github.com/david-yoon/attentive-modality-hopping-for-SER
Once you have it downloaded, replace the 'data_path' in 'multi_run.sh' with your folder path.

Note: The processed dataset repo from Dr. David Yoon contains, for each fold, '_nlp_trans.npy' and 'W_embedding.npy'. The former contains word tokens and the latter contains feature vectors for each token. Use the script pre_process_text_data.py to create the final '_text.npy' using both.

# Code
Use the bash file 'multi_run.sh' to run the 5 fold cross validation with 10 runs on each fold. Remember to do 'chmod +x ./multi_run.sh' before executing the bash file.

# Citation
If you use the sample code or part of it in your research, please cite the following:

```
@inproceedings{Cross_Self_Attn_Rajan_Cavallaro_2022,
       author = {{Rajan}, V. and {Brutti}, A. and {Cavallaro}, A.},
       title = "{Is cross-attention preferable to self-attention for multi-modal emotion recognition?}",
       booktitle = {Proceedings of the International Conference on Acoustics, Speech, and Signal Processing},
       pages={1--1},
       year = {2022},
       month = {May},
       organization={IEEE}
}
```
