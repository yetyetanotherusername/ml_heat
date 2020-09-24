import zarr


def main():
    store = zarr.DirectoryStore('ml_heat/__data_store__/zarr_store/store.zarr/origin')
    array = zarr.open(store, mode='r')

    for myslice in array.islice(1000000000, 1000000005):
        print(myslice)


if __name__ == '__main__':
    main()
