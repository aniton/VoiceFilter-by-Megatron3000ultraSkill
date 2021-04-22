# :mega: VoiceFilter by Megatron3000ultraSkill
This is Skoltech ML Final Project, devoted to
+ Separation of one speaker's voice from noise and other voices
+ Removing speaker's voice from the background noise

### Tasks
+ Replicate the resutls of [VoiceFilter paper](https://arxiv.org/pdf/1810.04826.pdf) :heavy_check_mark:
+ Analyze if filtered audio samples improve quality for downstream tasks, such as Speech-to-Text and Voice Gender Detection :heavy_check_mark: [here](https://github.com/aniton/VoiceFilter-by-Megatron3000ultraSkill/tree/main/downstream_tasks)
+ Reverse the model so it keeps all but one :heavy_check_mark:

### :rocket: Launch the model
#### Requirements
We tested the code on Python 3.6 with PyTorch 1.0.1. Other packages can be installed by:
  <pre> pip install -r requirements.txt</pre>
#### Datasets
+ For Speaker Encoder we used [VoxCeleb2 dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html).
+ For training and testing phases of Voicefilter we used [VoxCeleb1 datasets](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) <br> We selected for training all the utterances from 30 speakers, for testing all the utterances from 10 speakers. <br>

Both datasets consist of celebrities utterances, extracted from videos uploaded to YouTube.

#### Data Preparement
Perform STFT for train and test files before training by:
  <pre>python generator.py -c [config.yaml file] -d [VoxCeleb1 directory (should ends with <i>aac</i>)] -o [output directory]</pre>
We got 100,000 train and 1000 test audio samples (each of 3 sec).
#### The Model

![GitHub Logo](/model.png)

The model consists of
+ Speaker Encoder (3-layer LSTM), which produces a speaker embedding from audio samples of the target speaker
+ VoiceFilter (we used the variant with 8 convolutional layers, 1 LSTM layer, and 2 fully connected layers, each with ReLU activations except for the last layer, which has a sigmoid activation).

To **reimplement** the model run:
  <pre>python trainer.py -c [config.yaml file] -e [path of embedder pt file] -m [create a name for the model]</pre>

To get **the results**  run:
<pre> python res.py -c [config.yaml file] -e [path of embedder pt file] --checkpoint_path [chkpt/name/chkpt_{step}.pt] </pre>

#### Results
After 1.2k steps we got the following results:
![GitHub Logo](/res.png)

**See an example** :small_red_triangle_down: <br> https://aniton.github.io/videos.github.io/ <br>
Cut vesion of resulted samples: :small_red_triangle_down: <br> https://drive.google.com/file/d/1rEqKNUC7ZBb4MIJ5hgqrgYI_Ir6WRyyw/view?usp=sharing


Metrics comparing to **paper results**, where [LibriSpeech dataset](https://www.openslr.org/12) was used:

| Median SDR (Source to Distortion Ratio)  | Paper | Ours |
| ------------- | ------------- |------------- |
| Before VoiceFilter | 2.5  | 1.6 |
| After VoiceFilter  | 12.6 | 4.8 |


### :left_right_arrow: Reverse Task
For the task of keeping background noise and voices, but removing the target speaker voice we used the same system (Speech Embedder + Voicefilter).
The main difference was in **inputs**:
+ as Reference Audio and Clean Audio we used background noise
+ as Noisy Audio we used background noise with target speaker voice

For this purpose, we made use of [VOICES](https://iqtlabs.github.io/voices/Lab41-SRI-VOiCES_README/) dataset, which contains audio samples of clean speech from [Librispeech dataset](https://www.openslr.org/12) played with different background noise.

We generated train/test target and mixed audio samples by:
<pre>python rev_data_gen.py -p [path to VOiCES_devkit data] -s [path to save dir] -n [number generated samples by user]</pre>

We generated spectrograms for target and mixed audio samples by:
<pre>python rev_spec_gen.py -p [path to folder with train/test data] -s [path to save dir] -с [path to config.yaml file]</pre>

Then we trained the model in the same maner as in the first task.
#### Results
After 2k steps:
| Median SDR (Source to Distortion Ratio)  | Ours |
| ------------- |------------- |
| Before VoiceFilter |  1.8 |
| After VoiceFilter  | 5.9 |

**See an example** :small_red_triangle_down: <br> https://aniton.github.io/videos1.github.io/ <br>
Cut vesion of resulted samples: :small_red_triangle_down: <br> https://drive.google.com/file/d/14xXaB1WuUq9yqIT88L1ldKyh1SIf1tK9/view?usp=sharing

### Downstream tasks
Go to folder downstream_tasks and follow the README file there.


### Acknowledgements
Based on https://github.com/mindslab-ai/voicefilter and https://github.com/HarryVolek/PyTorch_Speaker_Verification.
### Team Members
+ [Mikhail Filitov](https://github.com/lll-phill-lll)
+ [Yaroslav Pudyakov](https://github.com/boomland)
+ [Ildar Gabdrakhmanov](https://github.com/KotShredinger)
+ [Anita Soloveva](https://github.com/aniton)
