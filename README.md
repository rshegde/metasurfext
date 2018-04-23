
# Metasurfext
These algorithms are developed for designing and optimization of beam deflectors.It was written for the purpose of doing calculations described the paper "Rapid design of wide-area heterogeneous electromagnetic metasurfaces beyond the unit-cell approximation" at doi:10.2528/PIERM17070405. 

Using the given 

## General set up
- Install python 3.4 or later version: https://anaconda.org/anaconda/python
- Download and install S4 (RCWA based software): https://web.stanford.edu/group/fan/S4/install.html
- Install Lua 5.2 or latest version: https://www.lua.org/download.html
- Keep grating.py, grating.lua, S4 and algorithm file in a same folder

## Using the code
Now, to optimize beam deflector,

The first step is to set the input parameters. The code is genetic algorithm.py or artificial bee colony.py which calls grating.py (which in turn calls grating.lua, the script that directly interfaces with S4). 

