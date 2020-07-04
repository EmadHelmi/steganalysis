# Steganalysis in python with deep learning algorithms
This project consists of two major parts:
+ Steganography
+ Steganalysis
## Steganography
In this part I have downloaded matlab codes from [Jessica Fridrich website](http://www.ws.binghamton.edu/fridrich/) (you can also get the algorithms [here](http://dde.binghamton.edu/download/stego_algorithms/)).

For embedding process I used these datasets:
+ BSDS300 ([Download](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz))
+ BSDS500 ([Download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz))
+ INRIA-jpg1 ([Download](http://lear.inrialpes.fr/people/jegou/data.php))
+ INRIA-jpg2 ([Download](http://lear.inrialpes.fr/people/jegou/data.php))

with **0.1**, **0.2**, **0.4** and **0.8** payloads. I use many embedding algorithms like **wow**, **J-Uniward** and **S-Uniward** which you can find their codes [here](http://dde.binghamton.edu/download/stego_algorithms). I put all codes in the [**graphy**](https://github.com/EmadHelmi/steganalysis/tree/master/graphy) folder.

## Steganalysis

I want to use two main models for steganalysis which are presented in these papers:

1. Yenet (with pytorch) [The publication can be found here](http://ieeexplore.ieee.org/document/7937836/).
2. Catalyst Kernels (with Keras and Tensorflow as its backend) [The publication can be found here](https://link.springer.com/chapter/10.1007/978-3-319-97749-2_9).

I put all codes in the [**analysis**](https://github.com/EmadHelmi/steganalysis/tree/master/analysis) folder.

**NOTE**: Currently I am developing the second one and I don't make any changes on the first model.
