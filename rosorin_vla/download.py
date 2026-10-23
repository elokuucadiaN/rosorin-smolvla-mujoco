"""Download and verify the versioned model release, without server credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
import urllib.request

REPOSITORY='elokuucadiaN/rosorin-smolvla-mujoco'
MANIFEST=Path(__file__).with_name('release_manifest.json')

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def verify_model(folder,manifest):
    for name,expected in manifest['model_files'].items():
        path=folder/name
        if not path.is_file() or sha256(path)!=expected:
            raise ValueError('MODEL_FILE_CHECKSUM_MISMATCH: '+name)

def extract_model(archive,output,manifest):
    output.mkdir(parents=True,exist_ok=True)
    destination=output/'smolvla-block'
    if destination.exists():
        verify_model(destination,manifest)
        return destination
    with tempfile.TemporaryDirectory(prefix='.extract-',dir=output) as temporary:
        root=Path(temporary).resolve()
        with tarfile.open(archive,'r:gz') as tar:
            for member in tar:
                path=(root/member.name).resolve()
                if not path.is_relative_to(root/'smolvla-block'):
                    raise ValueError('UNSAFE_ARCHIVE_PATH')
                if member.isdir():path.mkdir(parents=True,exist_ok=True)
                elif member.isfile():
                    relative=path.relative_to(root/'smolvla-block').as_posix()
                    if relative not in manifest['model_files']:raise ValueError('UNEXPECTED_MODEL_FILE')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(member) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst)
                else:raise ValueError('ARCHIVE_LINKS_NOT_ALLOWED')
        verify_model(root/'smolvla-block',manifest)
        (root/'smolvla-block').replace(destination)
    return destination

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'models')
    parser.add_argument('--asset',type=Path,help='Use an already downloaded release archive')
    args=parser.parse_args()
    manifest=json.loads(MANIFEST.read_text())
    destination=args.output/'smolvla-block'
    if destination.exists():
        verify_model(destination,manifest);print('Model already verified: '+str(destination));return
    item=manifest['assets'][0]
    archive=args.asset
    if archive is None:
        cache=args.output/'.downloads';cache.mkdir(parents=True,exist_ok=True)
        archive=cache/item['name']
        if not archive.exists() or sha256(archive)!=item['sha256']:
            url=f"https://github.com/{REPOSITORY}/releases/download/{manifest['version']}/{item['name']}"
            partial=archive.with_suffix('.part');last=time.monotonic()
            print('Downloading '+url,flush=True)
            with urllib.request.urlopen(url,timeout=60) as response,partial.open('wb') as stream:
                count=0
                while block:=response.read(4*1024*1024):
                    stream.write(block);count+=len(block)
                    if time.monotonic()-last>10:
                        print(f"{count/1e6:.0f}/{item['size_bytes']/1e6:.0f} MB",flush=True);last=time.monotonic()
            if sha256(partial)!=item['sha256']:raise ValueError('ARCHIVE_SHA256_MISMATCH')
            partial.replace(archive)
    if sha256(archive)!=item['sha256']:raise ValueError('ARCHIVE_SHA256_MISMATCH')
    print('Extracting verified model...',flush=True)
    print('Ready: '+str(extract_model(archive,args.output,manifest)),flush=True)

if __name__=='__main__':main()
