import zarr
from itertools import islice


def main():
    store = zarr.DirectoryStore('/data/mlheat/testset')
    array = zarr.open(store, mode='r')

    for myslice in islice(array, 0):
        print(myslice)


if __name__ == '__main__':
    main()
