
# About

This started as a CNN model for transcription factor binding prediction in DNA. Ported to PyTorch from the Jax implementation of chapter 3 of "Deep Learning for Biology" (Ravarani et al). The book informs that this task is inspired by one of the evaluation challenges presented in [1] and which sourced the dataset from [2].

In this version, a basic two layer CNN with batch normalization is trained on 10 transcription factors binding datasets. The mean AUC is calculated and written to metrics.

We then use a modified program.md from Andrej Karpathy's [autoresaerch project](https://github.com/karpathy/autoresearch) to instruct an LLM to make sequential improvements to the model to improve the mean validation auc (mean auc for all 10 models). 

And therefore also the training loop runs for a fixed number of seconds (defined in constants.py) rather than a fixed number of training steps. 

# Usage

- source a venv and install requirements. 

- python download.py  # downloads the dataset from the deeplearningforbiology website

- check if the base model training works:

```
#this trains 10 models using the same , produces loss logs in ./metrics
python main.py --op train 

# run prediction, produces auc validation metrics json in ./metrics
python main.py --op predict
```

- Kick off your dev agent with confirmations disabled such as

```
claude -p "Hi have a look at program.md and let's kick off a new experiment! let's do the setup first." --dangerously-skip-permissions
```

Of course run this on a sandbox if using that flag. I use https://ResearchBox.ai


# References

[1] Evaluating the representational power of pre-trained DNA language models for regulatory genomics. Ziqi Tang, Nirali Somia, Yiyang Yu, Peter K Koo (2024). Biorxiv doi: https://doi.org/10.1101/2024.02.29.582810  Now published in Genome Biology doi: 10.1186/s13059-025-03674-8

[2] Majdandzic, A., Rajesh, C. & Koo, P.K. Correcting gradient-based interpretations of deep neural networks for genomics. Genome Biol 24, 109 (2023). https://doi.org/10.1186/s13059-023-02956-3

