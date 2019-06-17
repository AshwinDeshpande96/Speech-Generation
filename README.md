# Speech-Generation
Textual Speech generation using LSTM network

We test several popular architecture for language modelling and test for word-prediction task.
* Vanilla LSTM with 100 context words
* Bidirectional LSTM with 10 context words


## 1. Dataset

Dataset used is the 'Corpus of Presidential Speeches' by Grammer Lab. Link: [Dataset](http://www.thegrammarlab.com/?nor-portfolio=corpus-of-presidential-speeches-cops-and-a-clintontrump-corpus#)

The dataset consist of 43 sets of presidential speeches for 43 different Presidents of the USA.
1.  coolidge's  12 speeches
2.  tyler's  18 speeches
3.  wilson's  32 speeches
4.  ford's  14 speeches
5.  pierce's  15 speeches
6.  lincoln's  15 speeches

.

.

.

41.  grant's  32 speeches
42.  jqadams's  8 speeches
43.  jackson's  26 speeches

### 1.1. Data Collection

We use Lyndon B. Johnson's speeches as it has 71 speeches (with 71 individual .txt files). We concatenated these 71 files into a single .txt file. This summed up to a 2.82M words and a vocabulary size of 8703. Developing a Speech Generator on this requires huge amount of memory. Google Colaboratory provides 12GB VRAM on Google's NVIDIA K80 powered GPU runtime. Output vector which would be of shape: (2.82M, >8703) is not suitable for language modelling tasks. Filtering words and inflection bring down the size to (1.34M, 8703).

Furthermore, after data pre-processing the Vanilla LSTM Model has a bottleneck at the Softmax Layer (Output Layer) due its size: O(|V|) and slows down training. This is addressed using different variation of probability estimation method.  

Our Project: [Hierarchical Softmax](https://github.com/AshwinDeshpande96/Hierarchical-Softmax) addresses this problem.

### 1.2. Data Pre-Processing
The text files contain several punctuation-symbols, numbers, spacings and word inflection. It is important to be careful and try to remove characters or letters such that it helps reduce the vocabulary size. Otherwise if vocabulary is large the numbers of classes increases. And the output layer will now have too many classes to predict. Large number of classes will slow down training and will require large resources and time to converge.
    
    text = "Today's weather condition is cloudy with a 76% of rain. Temperature may remain 
    cool at 21°C with Humidity 61%. Rainfall so far is measured at 130mm."
    
#### 1.2.1. Following punctuations have been removed:

    " ' . & , ? / : ; < > $ #  @ ! % * ( ) [ ] { } \n -
   
    
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    
   
Output:

            text = "Todays weather condition is cloudy with a 76 of rain  Temperature may remain 
    cool at 21 C with Humidity 61 Rainfall so far is measured at 130mm "

#### 1.2.2. Word Inflections are reduced to its root words:

  We use Porter Stemming or WordNet Lemmatization we brought down Vocabulary to 6673 or 8703 respectively.
  
    Output: 
            text = "today weath condit is cloudy with a seventy six of rain  temp may remain 
    cool at twenty on c with humid sixt on  rainfal so far is meas at on hundr and thirty mm "
  These words without their inflection make little sense, hence a dictionary variable is used where each root word entry is
  mapped to it's original inflection:
    
    dict = { today: Today's,
             weath: weather,
             condit: condition,
             is: is,
             cloudy: cloudy,
             with: with,
             .
             .
             .
             hundr: hundred,
             and: and,
             thirty: thirty,
             mm: mm }
   This however creates an issue: dict will have only one entry for a root word of different inflections.
   
This is substantially larger than it's character wise prediction counterpart. In comparison, the vocabulary for character prediction in worst case is as large as the ASCII character set: 256 characters. For natural language however vocabulary will consist of '[a-z, A-Z]' and punctuation. 
Prediction using character doesn't reflect human intelligence of forming sentences. Humans learn to form sentences by recognizing a word's contextual importance i.e. humans can infer a word's meaning depending on it's pattern of occurence.

n-gram models mimic this notion and calculate probabilities of next word succeeding a sequence based on it's prior occurence count. But, to predict a word w<sub>i</sub>, preceded by a sequence of contextual words: [w<sub>i-1</sub>,w<sub>i-2</sub> . . . w<sub>1</sub>], the probability may not always be contingent to it's prior occurence count. 
This approach has several disadvantages:
1. Predicted w<sub>i</sub> may only occur as a part of word pair.
        
      Given a dataset of news-media collected from San-Franisco Bay Area. It is natural that the word Fransisco occurs
        frequently in this text. Hence the probability P(w<sub>i</sub>='Fransisco') is very likely. Even though 'Fransisco' 
        will almost never occur without 'San', P(w<sub>i</sub>='Fransisco') will be most probable even if w<sub>i-1</sub> is 
        not 'San'. Vice-versa predicted P(w<sub>i</sub>='Fransisco') could be not likely if its frequency is low in a
        dataset, even if w<sub>i-1</sub> is 'San'.

2. When w<sub>i</sub> does not occur in dataset but is the correct word after w<sub>i-1</sub> (It's first occurence is encountered in the test set)

      Predicted w<sub>i</sub> will be one of |V| words with highest probability, even though w<sub>i</sub> was never
      encountered in training set. To counter this problem, some probability mass from vocabulary is subtracted and assigned
      to the new word. But, how much probability mass is to be given to new word? This cannot be estimated
      deterministically. 
      
However, several approaches such as back-off, interpolation and discounting techniques are proposed to mitigate these problems. Some of the popular ones are:
* Stupid back-off
* Katz's back-off model
* Good–Turing discounting
* Witten–Bell discounting
* Lidstone's smoothing
* Kneser–Ney smoothing

For more details on n-gram language models: 
* [INVESTIGATION OF BACK-OFF BASED INTERPOLATION BETWEEN
RECURRENT NEURAL NETWORK AND N-GRAM LANGUAGE MODELS](http://mi.eng.cam.ac.uk/~mjfg/asru15-chen.pdf)
* [Stanford NLP](https://www.youtube.com/watch?v=Saq1QagC8KY&list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm&index=12)

We can see that deterministic approach such as n-gram do not take into account the contextual information in dataset. It is not possible to manually design an algorithm to recognize occurence patterns. In order to build an algorithm to identify such patterns we need to understand what the logic is supposed to look for. Neither do we have a method to validate if found pattern is correct.

Recurrent Neural Network are best suited for this task. We treat the neural network as a black box, wherein said patterns are recognized such that sequential context data is taken into account. RNNs still fail to take into account the long-distance context dependencies, as they are sparsely distributed throughout the data.

Learn more about it here: [Yoshua Bengio - Presentation](https://www.bilibili.com/video/av34864474/), [Shortcomings of regular Encoder-Decoder Model(without Attention)](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/)

## 2. Model

* Vanilla LSTM network defined as follows: [Vanilla LSTM Network](https://github.com/AshwinDeshpande96/Speech-Generation/blob/master/president_NLP.ipynb)
![Image of LSTM](https://github.com/AshwinDeshpande96/Speech-Generation/blob/master/vanilla_LSTM.png)

<p align='center'> Fig-1: Vanilla LSTM (Network Definition) </p>

LSTM's ability to estimate sequential patterns has many applications such as:
* Gene Classification
* Speech Synthesis
* Music Generation
* Machine Translation

We can see one such implementation here: [Music Generation](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)

Network proposed here is suitable for vocabulary of smaller size. Even though the maximum computation workload is seen at Dense Layer and further Softmax scoring, networks with deeper LSTMs incur larger training time. Hence we use a network of single LSTM layer. This takes several days to train.

Next-Word prediction is not subject to overfitting. Hence, we use the entire dataset for training and none for validation. Because our goal here is to fit a model that behaves exactly like the dataset, out model is a high variance-low bias model. We only minimize the training error without validating across unseen data (Validation Set).

* Bidirectional LSTM network defined as follows: [biLSTM Network](https://github.com/AshwinDeshpande96/Speech-Generation/blob/master/biLM.ipynb)
![Image of biLSTM](https://github.com/AshwinDeshpande96/Speech-Generation/blob/master/bidirection_network.png)

<p align='center'> Fig-2: Bidirectional LSTM (Network Definition) </p>

This type of architecture produces a contextual representation of a word, which is extrememly important in NLP tasks such as:
* Word Sense Disambiguation
* Parts of Speech Tagging
* Named Entity Recognition
* Coreference Resolution

Furthermore learned embedding feature vectors form input for every other NLP tasks:
* Question Answering
* Textual Entailment
* Semantic Role Labelling
* Sentiment Analysis

![Image of biLSTM](https://github.com/AshwinDeshpande96/Speech-Generation/blob/master/biLM.jpeg)

<p align='center'> Fig-3: Bidirectional LSTM (Network Architecture) </p>

## 3. Results

Since RNNs are a highly sequential type of network, it is not very parallelizable. This property incurs huge training times. Our network was trained for approximately 7 days. 
