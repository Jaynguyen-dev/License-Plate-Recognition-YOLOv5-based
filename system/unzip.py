
#!/usr/bin/env python3
"""Unzip utility with tqdm progress bar.

Usage: python unzip.py archive.zip [dest_dir]
"""
import sys
import os
from zipfile import ZipFile
from tqdm import tqdm


def unzip_with_progress(zip_path, dest_dir=None):
	if dest_dir is None:
		dest_dir = os.path.splitext(os.path.basename(zip_path))[0]
	os.makedirs(dest_dir, exist_ok=True)

	with ZipFile(zip_path, 'r') as z:
		infos = z.infolist()
		total = sum(i.file_size for i in infos)

		with tqdm(total=total, unit='B', unit_scale=True, desc='Extracting') as pbar:
			for info in infos:
				# ensure target path is safe
				target_path = os.path.join(dest_dir, info.filename)
				if not os.path.realpath(target_path).startswith(os.path.realpath(dest_dir)):
					# skip files with malicious paths
					continue

				if info.is_dir():
					os.makedirs(target_path, exist_ok=True)
					continue

				# make sure directory exists
				os.makedirs(os.path.dirname(target_path), exist_ok=True)

				with z.open(info, 'r') as src, open(target_path, 'wb') as dst:
					# read in chunks and update progress
					for chunk in iter(lambda: src.read(1024 * 64), b''):
						dst.write(chunk)
						pbar.update(len(chunk))


def main(argv):
	if len(argv) < 2:
		print('Usage: unzip.py archive.zip [dest_dir]')
		return 1
	zip_path = argv[1]
	dest = argv[2] if len(argv) > 2 else None
	unzip_with_progress(zip_path, dest)
	return 0


if __name__ == '__main__':
	sys.exit(main(sys.argv))
