import itertools
from os import makedirs

def maybeCreateFolder(folder):
    makedirs(folder, exist_ok=True)


def progressiveFilenameGenerator(pattern="file_{}.ext"):
    for i in itertools.count():
        yield pattern.format(i)
