# Reconstructing-Perceptive-Images-from-Brain-Activity-by-Shape-Semantic-GAN
The codes of paper: "Reconstructing Perceptive Images from Brain Activity by Shape-Semantic GAN". 

Author : Tao Fang, Yu Qi, Gang Pan

This demo takes the published fMRI data "69-digits" as inputs.
This dataset contains 100 fMRI samples and 2 different stimulus images : '6' and '9'.
In this demo we take 90 samples as training set and 10 samples as test set.
To facilitate demonstration, the random seed is fixed to 0.

Directly run the script: 

"demo1_digits.py"

The reconstructed images will be saved in "results" fold.

<img src="https://github.com/duolala1/Reconstructing-Perceptive-Images-from-Brain-Activity-by-Shape-Semantic-GAN/blob/main/results/digits/dataset1_compare.png?raw=true" width="600" height="400" alt="Comparing part of the reconstructed samples with other methods visually."/><br/>


Comparing part of the reconstructed samples with other methods visually.


<img src="https://github.com/duolala1/Reconstructing-Perceptive-Images-from-Brain-Activity-by-Shape-Semantic-GAN/blob/main/results/digits/dataset1_compare_q.png?raw=true" width="400" height="300" alt="Comparing the reconstruction performance with other methods quantitatively."/><br/>


Comparing the reconstruction performance with other methods quantitatively.

The semantic decoder and shape decoder scripts are contained in the "model" fold.

Requirements:

Pytorch 1.7.0

