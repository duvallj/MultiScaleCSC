# Unsupervised Classification Challenge

Code to solve this challenge https://www.kaggle.com/yeayates21/garage-detection-unofficial-ssl-challenge?select=GarageImages

Algorithm to generate initial convolutional features adapted from the following paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5522776/

Still very much a work-in-progress, mainly meant to teach me how to use Tensorflow again

## TODO:
[ ] Implement algorithm used in paper to choose `Z` instead of just smacking more gradient descent on top of it
[ ] Actually use learned filters to classify images
[ ] Port code to use more Keras facilities for training

## Later TODO:
[ ] Compare performance against other Convolutional Neural Networks
[ ] Make visualization of kernels
