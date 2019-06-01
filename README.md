# Speech-Generation
Textual Speech generation using LSTM network

We have used a Recurrent Neural Network to generate Word Vectors from a Presidential Speech text.

## 1.1. Dataset
Dataset used is the 'Corpus of Presidential Speeches' by Grammer Lab. Link as follows:

http://www.thegrammarlab.com/?nor-portfolio=corpus-of-presidential-speeches-cops-and-a-clintontrump-corpus#

The dataset consist of 43 sets of presidential speeches for 43 different Presidents of the USA.
*  coolidge's  12 speeches
*  tyler's  18 speeches
*  wilson's  32 speeches
*  ford's  14 speeches
*  pierce's  15 speeches
*  lincoln's  15 speeches
*  washington's  21 speeches
*  reagan's  59 speeches
*  hoover's  29 speeches
*  jefferson's  24 speeches
*  bharrison's  16 speeches
*  monroe's  10 speeches
*  carter's  22 speeches
*  taft's  11 speeches
*  madison's  22 speeches
*  roosevelt's  22 speeches
*  eisenhower's  6 speeches
*  buchanan's  14 speeches
*  lbjohnson's  71 speeches
*  adams's  9 speeches
*  arthur's  11 speeches
*  fillmore's  7 speeches
*  kennedy's  45 speeches
*  fdroosevelt's  49 speeches
*  hayes's  16 speeches
*  obama's  48 speeches
*  bush's  23 speeches
*  johnson's  31 speeches
*  cleveland's  31 speeches
*  nixon's  23 speeches
*  harrison's  1 speeches
*  taylor's  4 speeches
*  clinton's  39 speeches
*  truman's  19 speeches
*  gwbush's  39 speeches
*  garfield's  1 speeches
*  harding's  18 speeches
*  mckinley's  14 speeches
*  vanburen's  10 speeches
*  polk's  25 speeches
*  grant's  32 speeches
*  jqadams's  8 speeches
*  jackson's  26 speeches

I initially tried using Lyndon B. Johnson's speeches as it has 71 speeches(With 71 individual .txt files). I concatenated these 71 files into a single .txt file. This summed up to a 2.46MM words and a vocabulary size of 9806. Developing a Speech Generator on this requires huge amount of memory. I tried executing on Google Colaboratory which provides 12GB VRAM on Google's NVIDIA K80 powered GPU runtime. Memory was insufficient for computing the one-hot vector because this vector will be of size (9806 x 9806).

Hence, I am using Abraham Lincoln's speeches with 15 text files. Concatenating these 15 speeches gives 1.01M words and a vocabulary size of |V| = 6308. Although this succeeded in obtaining the one-hot vector, it consumed a substantial portion of the memory. NLTK library for Lemmatization and Stemming is a handy tool for pruning the vocabulary.

Furthermore, after data pre-processing the Vanilla LSTM Model has a bottleneck at the Softmax Layer (Output Layer) due its size: O(|V|) and slows down training. This can addressed using different variations of Softmax Layer and different Output Layers altogether. Following are the Papers that help in this area.
* Strategies for Training Large Vocabulary Neural Language Models - (Chen, Grangier, Auli - 2015)
* Hierarchical Probabilistic Neural Network Language Model - (Morin, Bengio - 2005)
* A Scalable Hierarchical Distributed Language Model - (Mnih, Hinton - 2008)


<!--- An implementation of Hierarchical Softmax Layer will soon be published soon--->

## 1.2. Data Pre-Processing
The text files contain several punctuation-symbols, numbers, spacings and word inflection. It is important to be careful and try to remove characters or letters such that it helps reduce the vocabulary size. Otherwise if vocabulary is large the numbers of classes increases. And the output layer will now have too many classes to predict. Large number of classes will slow down training and will require large resources and time to converge.
    
    text = "Today's weather condition is cloudy with a 76% of rain. Temperature may remain 
    cool at 21°C with Humidity 61%. Rainfall so far is measured at 130mm."
    
### 1.2.1. Following punctuations have been removed with the exception of period:

    " & , ? / : ; < > $ #  @ ! % * ( ) [ ] { } \n -
   
   You can also choose to not remove other sentence breakpoints such as ! and ?. But doing so will include them as 
   words in the vocabulary. For the same reason period is considered as a word, as the position of periods are important for
   generated speech text to make sense.
   
   This is done using a simple python command and requires no extra libraries:
    
    text = text.replace(symbol, ' ')
    Output: 
            text = "Todays weather condition is cloudy with a 76 of rain . Temperature may remain 
    cool at 21 C with Humidity 61 . Rainfall so far is measured at 130mm ."
    
   These symbols are replaced by a space. Notice that we do not replace inverted commas by a space as word such as John's will
   obtain two words John and the letter s. Instead we replace ' with empty string so that John's --> Johns.
   If the text contains - it's and its - both, although they mean different it is treated as the same word. If it 
   appears in the generated text it is open to interpretation for the reader.

### 1.2.2. Numbers are replaced by their word form:
   First we need to find the numbers in the text. We do this using regular expressions.
    
    import re
    
   Following code returns a list of numbers in found in text. 
    
    num_set = re.findall(r'\d+', text)
    
    Output: 
            num_set : [76, 21, 61, 130]
   Now we have all the numbers in the text. We now convert into its word form. For this we use a Python Library - inflect.
   Code for this is as follows:
    
    p = inflect.engine()
    for num in num_set:
        word_form = p.number_to_words(num)
        text = text.replace(num, word_form)
    
    Output: 
            text = "Todays weather condition is cloudy with a seventy six of rain . Temperature may remain 
    cool at twenty one C with Humidity sixt one . Rainfall so far is measured at one hundred and thirty mm ."
   Numerical years will be written as 1976 is one thousand and seventy six rather than Nineteen Seventy Six.
### 1.2.3. Word Inflections are reduced to its root words:

  Using Lancaster Stemming and WordNet Lemmatization we brought down Vocabulary to 3737.
  
    Output: 
            text = "today weath condit is cloudy with a seventy six of rain . temp may remain 
    cool at twenty on c with humid sixt on . rainfal so far is meas at on hundr and thirty mm ."
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
        not 'San'. Vice-versa predicted P(w<sub>i</sub>='Fransisco') could be not likely if its frequency is low in a dataset, 
        even if w<sub>i-1</sub> is 'San'.
2. When w<sub>i</sub> does not occur in dataset but is the correct word after w<sub>i-1</sub> (It's first occurence is encountered in the test set)
        
      To counter this problem, some probability mass from vocabulary is subtracted and assigned to the new word. But, how 
        much probability mass is to be given to new word? This cannot be estimated deterministically. But several smoothing and back-off approaches have been proposed:
* Good–Turing discounting
* Witten–Bell discounting
* Lidstone's smoothing
* Stupid back-off
* Katz's back-off model
* Kneser–Ney smoothing
For more details on n-gram language models: 
* http://mi.eng.cam.ac.uk/~mjfg/asru15-chen.pdf
* https://web.stanford.edu/~jurafsky/slp3/3.pdf

We can see that deterministic approach such as n-gram do not take into account the contextual information in dataset. It is not possible to manually design an algorithm to recognize such patterns. In order to build an algorithm to identify such patterns we need to understand what the logic is supposed to look for. Neither do we have a method to validate if found pattern is correct.

Recurrent Neural Network are best suited for this task. We treat the neural network as a black box, wherein said patterns are recognized such that context data is taken into account. 
