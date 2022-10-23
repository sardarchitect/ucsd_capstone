import gzip 
import shutil
import tarfile

filepath = 'D:/00_SARDARCHITECTLABS/ucsd_capstone/datasets/399_buildings.csv.gz'
savepath = 'D:/00_SARDARCHITECTLABS/ucsd_capstone/datasets/399_buildings.csv'
savefolder = 'D:/00_SARDARCHITECTLABS/ucsd_capstone/datasets/'

def unzip(filepath, savepath):
    with gzip.open(filepath,'rb') as f_in:
        with open(savepath, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def untar(filepath, savefolder):
    file = tarfile.open(filepath)
    file.extractall(savefolder)
    file.close()

unzip(filepath, savepath)
