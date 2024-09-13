# Kirigami: Lightweight Speech Filtering for Privacy-Preserving Activity Recognition using Audio

[[paper (IMWUT 2024)](https://dl.acm.org/doi/10.1145/3643502)]

**Authors:**
[[Sudershan Boovaraghavan](https://sudershanb.com/)]
[[Haozhe Zhou](https://haozheee.github.io/)]
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
learning-based audio speech filter that removes human speech segments reducing the efficacy of various ASR techniques 
(70.48% PER and 101.40% WER) while also preventing sounds for HAR tasks from being filtered, thereby maintaining HAR 
accuracy (76.0% accuracy). We show that Kirigami can be implemented on common edge microcontrollers with limited 
computational capabilities and memory, providing a path to deployment on IoT devices. Finally, we conducted a real-world 
user study and showed the robustness of Kirigami on a laptop and an ARM Cortex-M4F microcontroller under three different 
background noises.
## Reference

[Download Paper Here](https://doi.org/)


BibTeX Reference:

```
@article{boovaraghavan2024kirigami,
  title={Kirigami: Lightweight speech filtering for privacy-preserving activity recognition using audio},
  author={Boovaraghavan, Sudershan and Zhou, Haozhe and Goel, Mayank and Agarwal, Yuvraj},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={8},
  number={1},
  pages={1--28},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```


## Installation:

### 1. Clone (or Fork!) this repository
```
git clone https://github.com/synergylabs/kirigami.git
```

### 2. Create a virtual environment and install python packages
We recommend using VirtualEnv. Tested on `Ubuntu 22.04`, with `python 3.10`.

```bash
python -m pip install -r requirements.txt
```

## Usage:
### Live Visualization
#### 1. Running the streamed visualization of Kirigami locally. 

```bash
python live_vis.py
```
### Filter Training:
#### 1. Download and prepare required dataset as mentioned in init_dataset.py. 
We will need:
- TIMIT (https://catalog.ldc.upenn.edu/LDC93S1)
- ESC-50 (https://github.com/karolpiczak/ESC-50)
- MS-SNSD (https://github.com/microsoft/MS-SNSD)

You need to download the TIMIT dataset yourself. The ESC-50 and MS-SNSD datasets will be downloaded automatically by the script.
```bash
python init_dataset.py
```
#### 2. Explore training your own Kirigami filters.
You can follow the steps inside experiments to train basic Logistic Regression models or customize your own models to detect speech.
- experiments/speech_detector.ipynb
- experiments/background_detector.ipynb

## Updates:
- 2024-09-13: Initial release of Kirigami live visualization and filter training scripts.

## Contact:
For more information please contact sudershan@cmu.edu or haozhezh@andrew.cmu.edu

