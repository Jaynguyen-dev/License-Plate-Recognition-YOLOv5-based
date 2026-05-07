
import os
import zipfile
from tqdm import tqdm
from pathlib import Path


def zip_folder(folder_path, output_zip, exclude=()):
	"""Zip a folder with a tqdm progress bar.

	Args:
		folder_path (str or Path): Path to folder to zip.
		output_zip (str or Path): Output zip file path.
		exclude (iterable): Iterable of filenames or directory names to exclude (basename match).
	"""
	folder_path = Path(folder_path)
	output_zip = Path(output_zip)

	# Gather all files
	files = []
	for root, dirs, filenames in os.walk(folder_path):
		# Optionally filter directories in-place so os.walk prunes them
		dirs[:] = [d for d in dirs if d not in exclude]
		for f in filenames:
			if f in exclude:
				continue
			files.append(Path(root) / f)

	# Create parent dir for output if needed
	output_zip.parent.mkdir(parents=True, exist_ok=True)

	with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
		for file_path in tqdm(files, desc=f"Zipping {folder_path.name}", unit="file"):
			# Compute archive name relative to the folder being zipped
			arcname = file_path.relative_to(folder_path.parent)
			zf.write(file_path, arcname)


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Zip a folder with progress bar")
	parser.add_argument("folder", help="Folder to zip")
	parser.add_argument("output", nargs="?", help="Output zip file (optional)")
	parser.add_argument("--exclude", "-e", nargs="*", default=[], help="Basename(s) to exclude")
	args = parser.parse_args()

	folder = Path(args.folder)
	if not folder.exists() or not folder.is_dir():
		raise SystemExit(f"Folder not found: {folder}")

	out = Path(args.output) if args.output else folder.with_suffix(".zip")
	zip_folder(folder, out, exclude=args.exclude)


if __name__ == "__main__":
	main()
