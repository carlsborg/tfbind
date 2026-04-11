
# About

A PyTorch CNN model for transcription factor binding prediction in DNA. Ported to PyTorch from the Jax implementation of chapter 3 of "Deep Learning for Biology" (Ravarani et al).  This text informs that this task is inspired by one of the evaluation challenges presented in [1] and which sourced the dataset from [2]

[1] Evaluating the representational power of pre-trained DNA language models for regulatory genomics. Ziqi Tang, Nirali Somia, Yiyang Yu, Peter K Koo (2024). Biorxiv doi: https://doi.org/10.1101/2024.02.29.582810  Now published in Genome Biology doi: 10.1186/s13059-025-03674-8

[2] Majdandzic, A., Rajesh, C. & Koo, P.K. Correcting gradient-based interpretations of deep neural networks for genomics. Genome Biol 24, 109 (2023). https://doi.org/10.1186/s13059-023-02956-3


# Usage

- source a venv and install requirements. 

- python download.py  # downloads the dataset from the deeplearningforbiology website

- train
  // this trains 10 models using the same , produces loss logs in ./metrics
 python main.py --op train 

- python main.py --op predict # run prediction , produces loss logs in ./metrics

