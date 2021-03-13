# :mega: VoiceFilter by Megatron3000ultraSkill 
This is Skoltech ML Final Project, devoted to 
+ Separation of one speaker's voice from noise and other voices 
+ Removing speaker's voice from the background noise

### Tasks
+ Replicate the resutls of [VoiceFilter paper](https://arxiv.org/pdf/1810.04826.pdf)
+ Analyze if filtered audio samples improve quality for speech recognition
+ Reverse the model so it keeps all but one

### :rocket: Launch the model 
We used **p2s.16xlarge.8** for the following scenario.
#### Requirements
We tested the code on Python 3.6 with PyTorch 1.0.1. Other packages can be installed by:
  <pre> pip install -r requirements.txt</pre>
#### Datasets
We used [LibriSpeech datasets](http://www.openslr.org/12/) for training: <code>train-clean-100.tar.gz</code>, for testing: <code>dev-clean.tar.gz</code>.

#### Resampling and Normalizing wav files
 <pre> tar -xvzf  train-clean-100.tar.gz <br>
 tar -xvzf  dev-clean.tar.gz</pre>
 After that we got two LibriSpeech folders. Further one should use only one LibriSpeech folder, where <code>train-clean</code> and <code>dev-clean</code> will be located (make necessary movements).
 <br>
 Copy <code>normalize-resample.sh</code> tgo this LibriSpeech folder and do the following:
  <pre>chmod a+x normalize-resample.sh <br>./normalize-resample.sh
</pre>
In <code>config/config.yaml</code> set train and test directories. <br>
Perform STFT for train and test files before training by:
  <pre>python generator.py -c [config.yaml file] -d [LibriSpeech directory] -o [output directory]</pre>
We got 100,000 train and 1000 test samples. 
#### The Model 

![GitHub Logo](/model.png)

The model consists of 
+ Speaker Encoder (3-layer LSTM), which produces a speaker embedding from audio samples of the target speaker 
+ VoiceFilter (we used the variant with 8 convolutional layers, 1 LSTM layer, and 2 fully connected layers, each with ReLU activations except for the last layer, which has a sigmoid activation). 

To **reimplement** the model run:
  <pre>python trainer.py -c [config.yaml file] -e [path of embedder pt file] -m [create a name for the model]</pre>
  
#### Results
After 450 steps we got the following results:
![GitHub Logo](/loss.png)

### Acknowledgements  
based on https://github.com/mindslab-ai/voicefilter.
### Team Members 
+ Mikhail Filitov 
+ Yaroslav Pudyakov
+ [Ildar Gabdrakhmanov](https://github.com/KotShredinger)
+ [Anita Soloveva](https://github.com/aniton)
