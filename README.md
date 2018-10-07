# FouCluster

*This project will be presented at [PyCon ES 2018](https://2018.es.pycon.org/).
An informative note can be found in spanish
[here](https://github.com/cperales/Fourier-classifying-songs/blob/master/PyConES_2018.md)*

## Motivation
Recommendation song systems nowadays, like **Spotify**, use song clustering by made up
[parameters](https://www.theverge.com/tldr/2018/2/5/16974194/spotify-recommendation-algorithm-playlist-hack-nelson)
such as *danceability*, *energy*, *instrumentalness*, ... etc, which need an expert in that area to create those
parameters.

In order to avoid expert knowledge and make access to machine
learning applied to song easier, this library
use signal analysis for measuring distances between songs.
With this distances, when the amount of songs is considerable clustering
can be applied.

Because [musical notes have associated frequencies](https://www.intmath.com/trigonometric-graphs/music.php),
this proposal is based on transforming from time series to frequency series, and then grouping theses series
using various techniques and distance metrics.

## Use

An example as a commented script, using this library, can be found in
[example.py](https://github.com/cperales/Fourier-classifying-songs/blob/master/example.py). Python
requirements are listed in *requirements.txt*, and it is also necessary install *mpg123* or *ffmpeg*.
