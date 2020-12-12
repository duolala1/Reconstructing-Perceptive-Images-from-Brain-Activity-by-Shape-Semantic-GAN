# Reconstructing-Perceptive-Images-from-Brain-Activity-by-Shape-Semantic-GAN
The codes of paper: "Reconstructing Perceptive Images from Brain Activity by Shape-Semantic GAN". 

This demo takes the published fMRI data "69-digits" as inputs.
This dataset contains 100 fMRI samples and 2 different stimulus images : '6' and '9'.
In this demo we take 90 samples as training set and 10 samples as test set.

Directly run the script: 
"demo1_digits.py"
The reconstructed images will be saved in "results" fold.

![Image text](https://github.com/duolala1/Reconstructing-Perceptive-Images-from-Brain-Activity-by-Shape-Semantic-GAN/blob/main/results/digits/reconstructed_img.png?raw=true)

The semantic decoder and shape decoder scripts are contained in the "model" fold.

Requirements:
Pytorch

