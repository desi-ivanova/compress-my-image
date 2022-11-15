import argparse
import pickle
from typing import Optional

from PIL import Image
import compressai
from utils import load_image, rgb_to_tensor, tensor_to_rgb, pad_image


def main(
    path_to_image: str,
    quality: int = 4,
    save_to_file: Optional[str] = None,
    verbose: bool = True,
):
    """
    Compress and decompress an image using a pretrained scale hyperprior model.

    Args:
        path_to_image: path to the image (should be png, jpeg or dng format)
        quality: quality level between 1 (lowest) and 8 (highest).
            Defaults to 4.
        save_to_file: if provided, the compressed image and meta data will be
            stored to <save_to_file> in pickle format.
        verbose: defaults to True.

    Returns:
        dictionary containing:
            compressed_data: output of running the compression method
            reconstruction: the reconstructed image (rgb)
            stats: total bits used to store the image (total_bits), bits per
                pixel (bpp), and mean squared error (mse) of the reconstruction
    """
    img_raw = load_image(path_to_image)
    orig_size = img_raw.shape[:2]
    img = rgb_to_tensor(img_raw)
    img = pad_image(img, orig_size, factor=16 * 16)

    # Load model
    hyperprior = compressai.zoo.bmshj2018_hyperprior(
        quality=quality, metric="mse", pretrained=True
    )
    hyperprior.eval()

    # Compress and calculate bit rate
    if verbose:
        print("Compressing...")
    compressed_data = hyperprior.compress(img)
    total_bits = 8 * (
        len(compressed_data["strings"][0][0]) + len(compressed_data["strings"][1][0])
    )
    bpp = total_bits / (orig_size[0] * orig_size[1])
    # Store the output if output file name is provided
    if save_to_file is not None:
        with open(save_to_file, "wb") as f:
            pickle.dump(compressed_data, f)
            print(f"Output saved to {save_to_file}")
    else:
        print("<save_to_file> not provided, output not saved!")

    if verbose:
        print(f"Image compressed! Total bits={total_bits}, bpp={bpp}.")
        print("Decompressing...", end="")

    # Decompress and reconstruct original rgb image
    decompressed = hyperprior.decompress(*compressed_data.values())
    mse = ((img - decompressed["x_hat"]) ** 2).mean()
    recon = tensor_to_rgb(decompressed["x_hat"])[: orig_size[0], : orig_size[1], :]
    if verbose:
        print(f"Mean squared error = {mse} ")
        print("Done :)")

    return {
        "compressed_data": compressed_data,
        "reconstruction": recon,
        "stats": {"total_bits": total_bits, "bpp": bpp, "mse": mse},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress my image")

    parser.add_argument("--image", default="test_image.dng", type=str)
    parser.add_argument("--quality", default=4, type=int)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-to-file", required=False, type=str)

    args = parser.parse_args()
    results = main(
        path_to_image=args.image, quality=args.quality, save_to_file=args.save_to_file
    )
    if args.plot:
        im = Image.fromarray(results["reconstruction"])
        im.show()
