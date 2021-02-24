# -*- coding: utf-8 -*-
import argparse
import os
import os.path
import pandas as pd
import subprocess
import shutil
import glob
import wget

def download_librispeech_subset(subset_name, download_dir):
    if os.path.exists(f'{download_dir}/{subset_name}.tar.gz'):
        print(f'{download_dir}/{subset_name} already exists - skipping download')
        return
    url = f'https://www.openslr.org/resources/12/{subset_name}.tar.gz'
    print(f'Downloading from {url} to {download_dir}/{subset_name}.tar.gz') 
    wget.download(url, out=download_dir)

def extract_subset(subset_name, download_dir, extracted_dir):
    if os.path.exists(f'{extracted_dir}/LibriSpeech/{subset_name}'):
        print(f'{extracted_dir}/LibriSpeech/{subset_name} already exists, skipping extraction')
        return
    os.makedirs(args.extracted_dir,exist_ok=True)
    print('Unpacking .tar file')
    shutil.unpack_archive(f'{download_dir}/{subset_name}.tar.gz', extracted_dir)

def read_transcriptions(subset_name,extracted_dir):
    all_lines = []
    transcript_glob = os.path.join(args.extracted_dir,'LibriSpeech',subset_name,'*/*/*.txt')
    print(transcript_glob)
    for transcript_file in glob.glob(transcript_glob):
        with open(transcript_file,'r') as f:
            lines = f.readlines()
            lines = [line.split(' ',1) for line in lines]
            for line in lines:
                line[0] = os.path.join(os.path.dirname(transcript_file),line[0]+'.flac')
        all_lines.extend(lines)
    return all_lines

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Librispeech data preparation')
    parser.add_argument('--subset',type=str,default='dev-clean',help='Subset of Librispeech to download.')
    parser.add_argument('--download_dir',type=str,default='.',help='Directory to download Librispeech to. Will be created if not exists.')
    parser.add_argument('--extracted_dir',type=str,default='./extracted', help='Directory to extract Librispeech to. Will be created if not exists.')
    parser.add_argument('--manifest_path',type=str,default='df.csv', help='Filename of the manifest to create. This is the path required by the "train.py" script')
    parser.add_argument('--use_relative_path',default=True,action='store_false',help='Use relative paths in resulting manifest.')
    args = parser.parse_args()
    
    progress_function = lambda gen : gen
    try:
        import tqdm
        progress_function = tqdm.tqdm
    except:
        print('tqdm not available, will not show progress')
    
    download_librispeech_subset(args.subset, args.download_dir)
    extract_subset(args.subset, args.download_dir, args.extracted_dir)
    all_lines = read_transcriptions(args.subset, args.extracted_dir)
    
    #os.makedirs(args.,exist_ok=True)
    
    df = pd.DataFrame(all_lines,columns=['audio_filepath','text'])
    if not args.use_relative_path:
        df.filepath = df.filepath.apply(os.path.abspath)
    df.text = df.text.apply(str.strip)
    df.to_csv(args.manifest_path)
    print(f'Done - manifest created at {args.manifest_path}')