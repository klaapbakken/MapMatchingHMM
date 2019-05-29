# MapMatchingHMM

Map Matching using Hidden Markov Models.

The data used to represent road networks is obtained from OpenStreetMap.

A route is simulated together with related observations. The observations are intended to resemble GPS - measurements and signals received from objects such as WiFi Access Points and cellphone towers. 

The conditional probabilties of the state sequences given observations are estimated using the Forward-Backward algorithm. The MAP sequence of segments is found using Viterbi.

The method can be tested by calling

```
python run_single_remotely.py
```

Requirements are listed in requirements.txt. Installation instructions (using conda) can be seen at the top of the file.
Use ``pip install utm`` and ``pip install osmapi`` to get remaining dependencies.

Please note that this is a work in progresss.
