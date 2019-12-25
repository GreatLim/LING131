# LING131 Report
**Report PDF version**: [Click here](https://github.com/GreatLim/LING131/blob/master/Ling131_Report.pdf)

**Author**: Mingzhen LIn & Yulun Wu


## HMM POS tagger

For training supervised HHM POS tagger, we can type in command like this.

```
$ python3 pos_tagger.py --train
```
And the output information would include the elapse time the accuracy.

```
Creating HHM POS tagger in ./pos_tagger.jbl
        Elapsed time: 5.205424785614014s
        Accuracy: 0.8967249002017427
```

For Running supervised HHM POS tagger, we can type in command like this.

```
$ python3 pos_tagger.py --run "Lets all be unique together until we realise we are all the same."
```

Given the input text, we can get the pairs of word and pos.

```
$ python3 pos_tagger.py --run "Lets all be unique together until we realise we are all the same."
* input text: Lets all be unique together until we realise we are all the same.
* pairs of word and pos: 
        Lets -- IN
        all -- ABN
        be -- BE
        unique -- JJ
        together -- RB
        until -- CS
        we -- PPSS
        realise -- MD
        we -- PPSS
        are -- BER
        all -- ABN
        the -- AT
        same -- AP
        . -- .
```

## Unsupervised POS tagger

For running unsupervised HHM POS tagger, we can type in command like this.

```
$python3 forward_backward.py
```

It would take dataset ﬁle as training data.

```
* Dataset: 
	he saw a cat
	a cat saw him
	he chased the cat
	the cat chased him
	he saw the dog
	the dog saw him
	he chased a dog
	a dog chased him
```

The result is,

```
* POS dict: {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
* Word dict: {0: 'he', 1: 'chased', 2: 'dog', 3: 'cat', 4: 'a', 5: 'saw', 6: 'the', 7: 'him'}
* Initial probability distribution: [[1.05076107e-26 2.15655953e-28 3.99517315e-30 5.71428571e-01
  5.71428571e-01]]
* Transition matrix: [[5.e-324 5.e-324 1.e+000 5.e-324 5.e-324]
 [5.e-324 5.e-324 5.e-324 1.e+000 5.e-324]
 [5.e-324 5.e-324 1.e+000 5.e-324 5.e-324]
 [1.e+000 5.e-324 5.e-324 5.e-324 5.e-324]
 [5.e-324 1.e+000 5.e-324 5.e-324 5.e-324]]
* Emission matrix: [[4.9e-324 4.9e-324 5.0e-001 5.0e-001 4.9e-324 4.9e-324 4.9e-324 4.9e-324]
 [4.9e-324 5.0e-001 4.9e-324 4.9e-324 4.9e-324 5.0e-001 4.9e-324 4.9e-324]
 [4.9e-324 2.5e-001 4.9e-324 4.9e-324 4.9e-324 2.5e-001 4.9e-324 5.0e-001]
 [4.9e-324 4.9e-324 4.9e-324 4.9e-324 5.0e-001 4.9e-324 5.0e-001 4.9e-324]
 [1.0e+000 4.9e-324 4.9e-324 4.9e-324 4.9e-324 4.9e-324 4.9e-324 4.9e-324]]
* Result:
	he/E saw/B a/D cat/A
	a/D cat/A saw/C him/C
	he/E chased/B the/D cat/A
	the/D cat/A chased/C him/C
	he/E saw/B the/D dog/A
	the/D dog/A saw/C him/C
	he/E chased/B a/D dog/A
	a/D dog/A chased/C him/C
```
