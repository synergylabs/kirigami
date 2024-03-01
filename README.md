# Kirigami: Lightweight Speech Filtering for Privacy-Preserving Activity Recognition using Audio

[[paper (IMWUT 2024)](https://doi.org/10.1145/3610896)]
[[talk (IMWUT 2024)](https://www.youtube.com/)]
[[demo video](https://www.youtube.com/)]

**Authors:**
[[Sudershan Boovaraghavan](https://sudershanb.com/)]
[[Haozhe Zhou](http://prasoonpatidar.com/)]
[[Mayank Goel](https://www.mayankgoel.com/)]
[[Yuvraj Agarwal](https://www.synergylabs.org/yuvraj/)]

**Abstract:**
Audio-based human activity recognition (HAR) is very popular due to many human activities having unique sound signatures 
that can be detected by machine learning (ML) approaches under various environmental contexts without re-training. These 
audio-based ML HAR pipelines often use common featurization techniques, such as extracting various statistical and 
spectral features by converting time domain signals to the frequency domain (using an FFT) and building ML models on top 
of them. Some of these approaches also claim privacy benefits by preventing the identification of human speech. However, 
recent deep learning-based automatic speech recognition (ASR) models pose new privacy challenges to these featurization 
techniques. In this paper, we aim to systematically characterize various proposed featurization approaches for audio 
data in terms of their privacy risks using a set of metrics for speech intelligibility (PER and WER) as well as the 
utility tradeoff in terms of the resulting accuracy of ML-based activity recognition. Our results demonstrate the 
vulnerability of several of these approaches to recent ASR models, particularly when subjected to re-tuning or 
retraining, with fine-tuned ASR models achieving an average Phoneme Error Rate (PER) of 39.99% and Word Error Rate 
(WER) of 44.43% in speech recognition for these approaches. We then propose Kirigami, a lightweight machine 
learning-based audio speech filter that removes human speech segments, reducing the efficacy of various ASR techniques 
(70.48% PER and 101.40% WER) while also preventing sounds for HAR tasks from being filtered, thereby maintaining HAR 
accuracy (76.0% accuracy). We show that Kirigami can be implemented on common-edge microcontrollers with limited 
computational capabilities and memory, providing a path to deployment on IoT devices. Finally, we conducted a real-world 
user study and showed the robustness of Kirigami on a laptop and an ARM Cortex-M4F microcontroller under three different 
background noises.
## Reference

[Download Paper Here](https://doi.org/)


BibTeX Reference:

```
@article{
}
```


## Installation:

### 1. Clone (or Fork!) this repository
```
git clone git@github.com:synergylabs/Kirigami-private-audio.git
```

### 2. Create a virtual environment and install python packages
We recommend using VirtualEnv. Tested on `Ubuntu 22.04`, with `python 3.10`.

```bash
python -m pip install -r requirements.txt
```

## Usage:

### 1. Running the Visualization locally. 

```bash

streamlit run main.py 
```

### 2. Using public website, go to 


## Contact:
For more information please contact sudershan@cmu.edu or haozhezh@andrew.cmu.edu

