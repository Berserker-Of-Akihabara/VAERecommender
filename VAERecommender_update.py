import requests
import zstandard as zstd
import tarfile
import gzip
import os
import datetime
import torch
import VAERecommender_hybrid as hy

def downloadDump(url, file_path):
    response = requests.get(url)
    if response.ok:
        file = open(file_path, "wb+")
        file.write(response.content)
        file.close()
    else:
        print("Failed to get the file")

def decompressTarZst(input_path, tar_path, final_folder):
    filesToGet = ['db/tags_parents', 'db/tags_vn', 'db/vn', 'db/users', 'db/tags', 'db/users', 'db/tags',\
                'db/releases', 'db/releases_lang', 'db/releases_vn']
    with open(input_path, 'rb') as compressed:
        decomp = zstd.ZstdDecompressor()
        with open(tar_path, 'wb') as destination:
            decomp.copy_stream(compressed, destination)
    tarFile = tarfile.open(tar_path)
    tarFile.extractall(path = final_folder, members=[tarFile.getmember(file) for file in filesToGet])
    tarFile.close()
    os.remove(input_path)
    os.remove(tar_path)

def decompresGZ(input_path, final_file, block_size = 65536):
    with gzip.open(input_path, 'rb') as s_file, \
            open(final_file, 'wb') as d_file:
        while True:
            block = s_file.read(block_size)
            if not block:
                break
            else:
                d_file.write(block)
        d_file.write(block)
    os.remove(input_path)

def updateModel(data, rec):
    print('UPDATE TIME')
    '''
    downloadDump('https://dl.vndb.org/dump/vndb-db-latest.tar.zst', 'dump.tar.zst')
    decompressTarZst('dump.tar.zst', 'dump.tar', './dump')
    downloadDump('https://dl.vndb.org/dump/vndb-votes-latest.gz', 'votes.gz')
    decompresGZ('votes.gz', './dump/db/votes')
    randomNumber = str(int(torch.randint(10000,50000,(1,))[0])) + datetime.datetime.today().strftime("%Y%m%d")
    '''
    print('downloading done')
    data_new = hy.RecommenderData(pathVN = './dump/db/vn', pathTags_vn = './dump/db/tags_vn',\
                                 pathTags_parents = './dump/db/tags_parents', pathTags= './dump/db/tags',
                                 pathVotes = './dump/db/votes', pathUsers='./dump/db/users',
                                 pathReleases= './dump/db/releases', pathReleasesLang='./dump/db/releases_lang',\
                                pathReleasesVN= './dump/db/releases_vn', pathLanguages='./dump/languages')
    print('data prepared')
    rec_new = hy.HybridRecommender(data_new, retrain = True)
    print('UPDATED')
    data, rec = data_new, rec_new

if __name__ == '__main__':
    data = None
    rec = None
    updateModel(data, rec)