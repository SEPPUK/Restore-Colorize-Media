import argparse
from os import listdir
from os.path import join, isfile, isdir

from PIL import Image
from resizeimage import resizeimage

from .shared import maybeCreateFolder


class ImageNetResizer:
    """Class instance to resize the images."""

    def __init__(self, sourceDir, destDir):
        """Constructor.

        Args:
            sourceDir (str): Path to folder containing all original images.
            destDir (str): Path where to store resized images.

        Raises:
            Exception: If sourceDir does not exist.

        """
        if not isdir(sourceDir):
            raise Exception("Input folder does not exist: {}".format(sourceDir))
        self.sourceDir = sourceDir

        # Destination folder
        maybeCreateFolder(destDir)
        self.destDir = destDir

    def resizeImg(self, filename, size=(299, 299)):
        """Resize image using padding.

        Resized image is stored in `destDir`.

        Args:
            filename (str): Filename of specific image.
            size (Tuple[int, int], optional): Output image shape. Defaults to (299, 299).

        """
        img = Image.open(join(self.sourceDir, filename))
        origWidth, origHeight = img.size
        wantedWidth, wantedHeight = size
        ratioW, ratioH = wantedWidth / origWidth, wantedHeight / origHeight

        enlargeFactor = min(ratioH, ratioW)
        if enlargeFactor > 1:
            # Both sides of the image are shorter than the desired dimension,
            # so take the side that's closer in size and enlarge the image
            # in both directions to make that one fit
            enlargedSize = (
                int(origWidth * enlargeFactor),
                int(origHeight * enlargeFactor),
            )
            img = img.resize(enlargedSize)

        # Now we have an image that's either larger than the desired shape
        # or at least one side matches the desired shape and we can resize
        # with contain
        res = resizeimage.resize_contain(img, size).convert("RGB")
        res.save(join(self.destDir, filename), res.format)

    def resizeAll(self, size=(299, 299)):
        """Resizes all images within `sourceDir`.

        Args:
            size (tuple, optional): Output image shape. Defaults to (299, 299).
        """
        for filename in listdir(self.sourceDir):
            imgPath = join(self.sourceDir, filename)
            if filename.endswith((".jpg", ".jpeg")) and isfile(imgPath):
                self.resizeImg(filename, size)


def _parseArgs():
    """Argparse setup.

    Returns:
        Namespace: Arguments.

    """

    def sizeTuple(size: str):
        size = tuple(map(int, size.split(",", maxsplit=1)))
        if len(size) == 1:
            size = size[0]
            size = (size, size)
        return size

    parser = argparse.ArgumentParser(
        description="Resize all images in a folder to a common size."
    )
    parser.add_argument(
        "core", type=str, metavar="SRC_DIR", help="Resize all images in SRC_DIR"
    )
    parser.add_argument(
        "output", type=str, metavar="OUT_DIR", help="Save resized images in OUT_DIR"
    )
    parser.add_argument(
        "-s --size",
        type=sizeTuple,
        default=(299, 299),
        metavar="SIZE",
        dest="size",
        help="Resize images to SIZE, can be a single integer or two comma-separated (W,H)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parseArgs()
    ImageNetResizer(sourceDir=args.core, destDir=args.output).resizeAll(
        size=args.size
    )
    print("Done")
