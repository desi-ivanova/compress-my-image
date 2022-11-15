This is a small, self-contained repo that lets you compress your own images using some of the latest and greatest (pretrained) neural compression algorithms.
It uses the excellent [compressai library](https://github.com/InterDigitalInc/CompressAI).


## Set-up

Clone the repo and create a conda environment by running

```bash
conda env create -f env.yml
conda activate compress
```

To run the example do

```bash
python3 compress_my_image.py --image=test_image.dng --quality=5 --plot
```

The model used to compress an input image is that of Ballé et al (2018) as implemented and trained in the [compressai library](https://github.com/InterDigitalInc/CompressAI).
The utility of this repo is in the data preparation (which is not hard but time-consuming -- it took me half a day to set up). This includes things like:

- loading an RGB image (8-bit integers) and converting it in the appropriate format (tensor of floats in [0, 1])
- ensure appropriate padding for the scale hyperprior architecture to work out
- calculating basic statistics (size, bits per pixel, distortion)
- converting the reconstruction back to RGB space
- conda environment setup etc.

Disclaimer: use at your own risk

### References
Ballé, J., Minnen, D., Singh, S., Hwang, S. J., & Johnston, N. (2018). Variational image compression with a scale hyperprior. ICLR. arXiv:1802.01436.

Bégaint, J., Racapé, F., Feltman, S., & Pushparaja, A. (2020). Compressai: a pytorch library and evaluation platform for end-to-end compression research. arXiv preprint arXiv:2011.03029.