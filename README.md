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

**Update, 5th of September 2019**: This project has been abandoned in favour of two other projects, [`tmmpy`](https://github.com/klaapbakken/tmmpy) and [`hmmpy`](https://github.com/klaapbakken/hmmpy). `hmmpy` aims to implement to common Hidden Markov Model functionality for arbitrary state spaces, observations, transition probabilities and emission probabilties. `tmmpy` leverages the functionality of `hmmpy` in order to do map matching. Both projects are already in a better state than this one, and will be worked on actively at least for a couple of months. I expect both projects to eventually end up on PyPI. 
