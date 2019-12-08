#LING131 Report

* Mingzhen LIn
* Yulun Wu

## HMM POS tagger

### Theory

### Usage
train
```
$ python3 pos_tagger.py --train
Creating Bayes classifier in ./pos_tagger.jbl
        Elapsed time: 5.205424785614014s
        Accuracy: 0.8967249002017427
```
run
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

## Todo
* Unsupervised POS tagger