# MapMatchingHMM

Map Matching using Hidden Markov Models

The method can be tested by calling 

```
python main.py <password> <desired action>
```

This requries that there exists a PostGIS database created using the Osmosis command-line application.
The name of the database should be "geodatabase" and the password should be <password>. 

If ``<desired action> == cache``, the script assumes that transition probabilites are saved to P.npy, 
and loads this instead of recomputing. 

The packages used will be posted at a later date. Python 3.5.6 has been used. 
