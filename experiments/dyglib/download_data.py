"""Download the official DyGLib benchmark archives without overwriting data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

DATASETS = ('wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo',
            'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts')
RECORD = 'https://zenodo.org/api/records/7213796'


@contextmanager
def member_stream(bundle, member):
    if member.compress_type == 9:  # Official Reddit archive uses Deflate64.
        if not shutil.which('unzip'):
            raise RuntimeError('Reddit requires unzip with Deflate64 support (install unzip)')
        with subprocess.Popen(['unzip', '-p', str(bundle.filename), member.filename], stdout=subprocess.PIPE) as process:
            try:
                yield process.stdout
            finally:
                process.stdout.close()
            if process.wait() != 0:
                raise RuntimeError(f'unzip failed for {member.filename}')
    else:
        with bundle.open(member) as stream:
            yield stream


def validate(directory, name):
    """Check the arrays/indexing expected by the native DyGLib loader."""
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(directory / f'ml_{name}.csv')
    edges = np.load(directory / f'ml_{name}.npy', mmap_mode='r')
    nodes = np.load(directory / f'ml_{name}_node.npy', mmap_mode='r')
    if not len(frame) or not {'u', 'i', 'ts', 'idx', 'label'}.issubset(frame.columns):
        raise ValueError(f'Invalid event columns: {name}')
    if not frame.ts.is_monotonic_increasing or not np.isfinite(frame.ts).all():
        raise ValueError(f'Invalid timestamp order: {name}')
    if not np.array_equal(frame.idx.to_numpy(), np.arange(1, len(frame)+1)):
        raise ValueError(f'Invalid edge indices: {name}')
    if min(frame.u.min(), frame.i.min()) < 1 or max(frame.u.max(), frame.i.max()) >= len(nodes):
        raise ValueError(f'Node IDs exceed feature rows: {name}')
    if edges.ndim != 2 or nodes.ndim != 2 or len(edges) != len(frame)+1 or max(edges.shape[1], nodes.shape[1]) > 172:
        raise ValueError(f'Incompatible feature dimensions: {name}')
    return {'dataset': name, 'events': len(frame),
            'nodes': len(np.union1d(frame.u, frame.i)),
            'edge_features': edges.shape[1], 'node_features': nodes.shape[1]}


def digest(path, algorithm='md5'):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/dyglib'))
    parser.add_argument('--target', type=Path, default=Path('derived/dyglib/processed_data'))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    manifest = args.cache / 'zenodo-7213796.json'
    if not manifest.exists():
        with urllib.request.urlopen(RECORD, timeout=120) as response:
            manifest.write_bytes(response.read())
    records = {entry['key']: entry for entry in json.loads(manifest.read_text())['files']}

    def fetch(name):
        entry = records[name + '.zip']
        archive = args.cache / entry['key']
        algorithm, checksum = entry['checksum'].split(':')
        if not archive.exists():
            partial = archive.with_suffix('.zip.partial')
            print(f'Downloading {name}: {entry["size"]/1e6:.1f} MB', flush=True)
            with urllib.request.urlopen(entry['links']['self'], timeout=180) as response, partial.open('wb') as output:
                shutil.copyfileobj(response, output, length=1024*1024)
            if partial.stat().st_size != entry['size'] or digest(partial, algorithm) != checksum:
                raise ValueError(f'Checksum/size mismatch: {partial}')
            partial.rename(archive)
        if archive.stat().st_size != entry['size'] or digest(archive, algorithm) != checksum:
            raise ValueError(f'Checksum/size mismatch: {archive}')
        # Copy only the three official processed files, never arbitrary archive paths.
        with zipfile.ZipFile(archive) as bundle:
            directory = args.target / name
            directory.mkdir(parents=True, exist_ok=True)
            for filename in (f'ml_{name}.csv', f'ml_{name}.npy', f'ml_{name}_node.npy'):
                matches = [x for x in bundle.infolist() if Path(x.filename).name == filename and '__MACOSX' not in x.filename]
                if len(matches) != 1:
                    raise ValueError(f'Expected one {filename} in {archive}, found {len(matches)}')
                dest = directory / filename
                if dest.exists():
                    # Verify existing files rather than silently accepting or replacing them.
                    h = hashlib.sha256()
                    with member_stream(bundle, matches[0]) as source:
                        for block in iter(lambda: source.read(1024*1024), b''):
                            h.update(block)
                    if digest(dest, 'sha256') != h.hexdigest():
                        raise ValueError(f'Refusing to overwrite different existing file: {dest}')
                else:
                    partial = dest.with_suffix(dest.suffix + '.partial')
                    with member_stream(bundle, matches[0]) as source, partial.open('wb') as output:
                        shutil.copyfileobj(source, output, length=1024*1024)
                    partial.rename(dest)
        print(f'Installed and verified {name}', flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, DATASETS))
    myket = args.target / 'myket'
    if not all((myket / f'ml_myket{suffix}').exists() for suffix in ('.csv', '.npy', '_node.npy')):
        raise FileNotFoundError('Myket should be supplied by the pinned DyGLib checkout; run setup first')
    report = []
    for name in (*DATASETS, 'myket'):
        item = validate(args.target / name, name)
        report.append(item)
        print(f'Validated {name}: {item["nodes"]:,} nodes, {item["events"]:,} events', flush=True)
    (args.cache / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('All 14 DyGLib datasets available (13 verified archives plus bundled Myket).', flush=True)


if __name__ == '__main__':
    main()
