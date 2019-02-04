# MapMatchingHMM

Map Matching using Hidden Markov Models.

A route is simulated together with related observations. The observations are intended to resemble GPS - measurements and signals received from things such WiFi Access Points and cellphone towers. 

The conditional probabilties of the state sequences given observations are estimated using the Forward-Backward algorithm. The MAP sequence of segments is found using Viterbi.

The method can be tested by calling

```
python main2_remote.py
```

Requirements are listed in requirements.txt. Installation instructions (using conda) can be seen at the top of the file.

Please note that this is a work in progresss.
