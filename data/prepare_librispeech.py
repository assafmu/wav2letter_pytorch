# -*- coding: utf-8 -*-
import argparse
import os
import os.path
import pandas as pd
import subprocess
import shutil
import glob
parser = argparse.ArgumentParser('Librispeech data preparation')
parser.add_argument('--zip_file',type=str,default='./dev-clean.tar.gz')
parser.add_argument('--extracted_dir',type=str,default='./dev-clean')
parser.add_argument('--target_dir',type=str,default='./dataset')
parser.add_argument('--manifest_path',type=str,default='df.csv')
parser.add_argument('--ffmpeg_path',type=str,default='ffmpeg')
parser.add_argument('--sample_rate',type=int,default=8000)
parser.add_argument('--use_abs_path',default=False,action='store_true',help='Use absolute paths in resulting manifest')
args = parser.parse_args()

progress_function = lambda gen : gen
try:
    import tqdm
    progress_function = tqdm.tqdm
except:
    print('tqdm not available, will not show progress')

if not os.path.exists(args.extracted_dir):
    os.makedirs(args.extracted_dir,exist_ok=True)
    print('Unpacking archive')
    shutil.unpack_archive(args.zip_file,args.extracted_dir)
	
def convert_to_wav(src,dest,ffmpeg_path,sr):
    if os.path.exists(dest):
        return
    conversion_command = r'%s -hide_banner -loglevel warning -i %s -ar %d %s' % (ffmpeg_path,src,sr,dest)
    subprocess.check_output(conversion_command.split(' '))


def get_wav_filename(src):
    return os.path.join(args.target_dir,os.path.splitext(os.path.basename(src))[0] + '.wav')


def convert_librispeech_audio(src):
    dest = get_wav_filename(src)
    convert_to_wav(src,dest,args.ffmpeg_path,args.sample_rate)

all_lines = []
transcript_glob = os.path.join(args.extracted_dir,'LibriSpeech',os.path.basename(args.extracted_dir),'*/*/*.txt')
for transcript_file in glob.glob(transcript_glob):
    with open(transcript_file,'r') as f:
        lines = f.readlines()
        lines = [line.split(' ',1) for line in lines]
        for line in lines:
            line[0] = os.path.join(os.path.dirname(transcript_file),line[0]+'.flac')
    all_lines.extend(lines)

os.makedirs(args.target_dir,exist_ok=True)

df = pd.DataFrame(all_lines,columns=['flac_filepath','text'])
print('Converting audio')
for flac in progress_function(df.flac_filepath):
    convert_librispeech_audio(flac)
df = df.assign(filepath=df.flac_filepath.apply(get_wav_filename))
if args.use_abs_path:
    df.filepath = df.filepath.apply(os.path.abspath)
df.text = df.text.apply(str.strip)
df.to_csv(args.manifest_path)
print('Done')