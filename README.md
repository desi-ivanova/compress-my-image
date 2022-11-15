This is a small, self-contained repo that lets you compress images using some of the latest and greatest (pretrained) neural compression algorithms. It uses the excellent [compressai library](https://github.com/InterDigitalInc/CompressAI).

## Set-up

Clone the repo and create a conda environment by running

```
conda env create -f env.yml
conda activate compress
```

To run the example do

```
python3 compress_my_image.py --image=test_image.dng --quality=5 --plot
```
