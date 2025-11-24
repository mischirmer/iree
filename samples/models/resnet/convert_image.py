"""
convert_image.py

This script prepares image inputs for ResNet by collecting up to `--limit`
images from either a local ImageNet directory or a text file with image URLs,
saving the original images into an `images/` subfolder and the binary input
files into a `bins/` subfolder. Each binary is a single-batch (1,224,224,3)
float32 file suitable for many ResNet examples.

Usage examples:
  - Use a local ImageNet folder as the source:
	  python convert_image.py --source-dir /path/to/imagenet/val --limit 100

  - Use a text file with image URLs (one URL per line):
	  python convert_image.py --urls-file imagenet_urls.txt --limit 100

If neither `--source-dir` nor `--urls-file` is provided the script will
attempt to download a small example image (same as before) and process it.
"""

import argparse
import io
import os
import shutil
from urllib.request import urlopen

from PIL import Image
import numpy as np


def ensure_dir(path):
	os.makedirs(path, exist_ok=True)


def clear_image_dir(images_dir):
	if not os.path.isdir(images_dir):
		return
	for f in os.listdir(images_dir):
		path = os.path.join(images_dir, f)
		if os.path.isfile(path) and is_image_file(f):
			try:
				os.remove(path)
			except Exception:
				pass


def is_image_file(name):
	name = name.lower()
	return name.endswith(('.jpg', '.jpeg', '.png', '.bmp'))


def copy_from_source(source_dir, images_dir, limit):
	files = [os.path.join(source_dir, f) for f in os.listdir(source_dir)]
	files = [f for f in files if os.path.isfile(f) and is_image_file(f)]
	files.sort()
	copied = 0
	for f in files:
		if copied >= limit:
			break
		dest = os.path.join(images_dir, os.path.basename(f))
		try:
			shutil.copyfile(f, dest)
			copied += 1
		except Exception as e:
			print(f"Skipping {f}: {e}")
	return copied


def download_from_urls(urls_file, images_dir, limit):
	downloaded = 0
	with open(urls_file, 'r') as fh:
		for line in fh:
			if downloaded >= limit:
				break
			url = line.strip()
			if not url:
				continue
			name = os.path.basename(url.split('?')[0]) or f'image_{downloaded}.jpg'
			dest_path = os.path.join(images_dir, name)
			try:
				data = urlopen(url, timeout=30).read()
				with open(dest_path, 'wb') as out:
					out.write(data)
				downloaded += 1
			except Exception as e:
				print(f"Failed to download {url}: {e}")
	return downloaded


def process_images(images_dir, bins_dir, resize=(224, 224)):
	files = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
	files = [f for f in files if os.path.isfile(f) and is_image_file(f)]
	files.sort()
	processed = 0
	for idx, f in enumerate(files, start=1):
		try:
			with Image.open(f) as img:
				img = img.convert('RGB')
				img = img.resize(resize, Image.BILINEAR)
				arr = np.array(img).astype(np.float32)
				arr = arr.reshape((1,) + arr.shape)  # (1,224,224,3)
				out_name = f'resnet_input_{idx}_{{}}_{{}}_{{}}_f32.bin'.format(
					resize[0], resize[1], arr.shape[-1]
				)
				out_path = os.path.join(bins_dir, out_name)
				arr.tofile(out_path)
				print(f"Wrote {out_path}")
				processed += 1
		except Exception as e:
			print(f"Skipping {f} (error: {e})")
	return processed


def main():
	parser = argparse.ArgumentParser(description='Prepare ImageNet images and binaries')
	parser.add_argument('--source-dir', help='Local directory containing ImageNet images')
	parser.add_argument('--urls-file', help='Text file with image URLs, one per line')
	parser.add_argument('--images-dir', default='images', help='Output images directory')
	parser.add_argument('--bins-dir', default='bins', help='Output binaries directory')
	parser.add_argument('--limit', type=int, default=100, help='Maximum number of images to process')
	parser.add_argument('--size', type=int, nargs=2, metavar=('W', 'H'), default=(224, 224), help='Resize W H')
	parser.add_argument('--use-tfds', action='store_true', help="Load a dataset from TensorFlow Datasets (won't auto-download unless --tfds-download used)")
	default_tfds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tfds_data')
	parser.add_argument('--tfds-dataset', default='cifar10', help='TFDS dataset name to load when --use-tfds is specified (default: cifar10)')
	parser.add_argument('--tfds-download', action='store_true', help='Allow tfds to download datasets (may attempt large downloads; not recommended)')
	parser.add_argument('--tfds-data-dir', default=default_tfds_dir, help='Directory TFDS should use for downloads and dataset storage (default: ./tfds_data in this samples folder)')
	parser.add_argument('--use-keras', action='store_true', help='Use Keras datasets (e.g., cifar10) instead of TFDS')
	args = parser.parse_args()
	total_collected = 0

	# If the user didn't pass a source, try to find a local ImageNet dataset
	def find_imagenet_dir():
		candidates = [
			os.environ.get('IMAGENET_DIR'),
			os.environ.get('DATASET_DIR'),
			'/data/imagenet',
			'/datasets/imagenet',
			'/mnt/imagenet',
			os.path.expanduser('~/imagenet'),
			'/imagenet/ILSVRC2012_img_val',
			'/data/imagenet/ILSVRC2012_img_val',
			'/datasets/imagenet/ILSVRC2012_img_val',
			'/data/imagenet/val',
			'/datasets/imagenet/val',
		]
		for c in candidates:
			if not c:
				continue
			c = os.path.expanduser(c)
			if os.path.isdir(c):
				# check for at least one image file
				for root, _, files in os.walk(c):
					for f in files:
						if is_image_file(f):
							return c
		return None

	# If no explicit source passed, attempt to locate ImageNet locally and use it.
	if not args.source_dir and not args.urls_file:
		imagenet_dir = find_imagenet_dir()
		if imagenet_dir:
			args.source_dir = imagenet_dir
			print(f"Detected ImageNet dataset at '{imagenet_dir}', using it as source (default).")
		else:
			print("No local ImageNet dataset detected; will fall back to example download if nothing collected.")

	# Optional: load from TensorFlow Datasets' ImageNet if requested.
	if args.use_keras and args.tfds_dataset == 'cifar10':
		try:
			from tensorflow.keras.datasets import cifar10
		except Exception:
			print('tensorflow.keras.datasets not available in this environment; install TensorFlow to use --use-keras')
			args.use_keras = False
		else:
			print('Loading CIFAR-10 via Keras datasets')
			ensure_dir(args.images_dir)
			clear_image_dir(args.images_dir)
			(x_train, y_train), (x_test, y_test) = cifar10.load_data()
			# use test set first, then train if more images needed
			images = np.concatenate([x_test, x_train], axis=0)
			count = 0
			for i in range(min(len(images), args.limit)):
				arr = images[i]
				pil = Image.fromarray(arr.astype('uint8'), 'RGB')
				name = f'keras_cifar10_{i+1}.png'
				pil.save(os.path.join(args.images_dir, name))
				count += 1
			print(f'Collected {count} images from Keras CIFAR-10 into "{args.images_dir}"')
			total_collected = count
	if args.use_tfds:
		try:
			import tensorflow_datasets as tfds
		except Exception:
			print("tensorflow_datasets is not installed. Install with: pip install tensorflow-datasets")
			args.use_tfds = False
		else:
			# Only use tfds if no other source was given.
			if not args.source_dir and not args.urls_file:
				download_flag = bool(args.tfds_download)
				if download_flag:
					print("Warning: --tfds-download set. TFDS may attempt large downloads for ImageNet and may still require manual tar files. Proceeding with download=True.")
				dataset_name = args.tfds_dataset
				# Choose a sensible default split depending on what's available.
				builder = tfds.builder(dataset_name, data_dir=args.tfds_data_dir)
				available_splits = list(builder.info.splits.keys())
				if 'test' in available_splits:
					split = 'test'
				elif 'validation' in available_splits:
					split = 'validation'
				elif 'train' in available_splits:
					split = 'train'
				else:
					# fallback to the first available split
					split = available_splits[0] if available_splits else None
				# Ensure TFDS uses the requested data directory.
				data_dir = os.path.abspath(args.tfds_data_dir)
				ensure_dir(data_dir)
				# set env var for TFDS as some code paths check it
				os.environ['TFDS_DATA_DIR'] = data_dir
				try:
					ds, info = tfds.load(dataset_name, split=split, with_info=True, shuffle_files=False, download=download_flag, data_dir=data_dir)
				except Exception as e:
					if download_flag:
						print("Attempt to download ImageNet via TFDS failed. You may need to place manual tar files into TFDS manual directory as described in the README or use --use-tfds without --tfds-download after preparing tfds.")
					else:
						print(f"Dataset {dataset_name} not available locally via tfds. To download, run: `python -m tensorflow_datasets.scripts.download_and_prepare --datasets={dataset_name}` or run with --tfds-download to attempt download.")
					print(f"(tfds error: {e})")
					args.use_tfds = False
				else:
					print(f"Loading images from tfds '{dataset_name}' split '{split}'")
					ensure_dir(args.images_dir)
					clear_image_dir(args.images_dir)
					count = 0
					for example in ds.take(args.limit):
						# tfds returns dicts with an 'image' field for supervised datasets
						if isinstance(example, dict) and 'image' in example:
							img_tensor = example['image']
						else:
							img_tensor = example
						try:
							arr = img_tensor.numpy()
						except Exception:
							try:
								import tensorflow as tf
								arr = tf.keras.backend.get_value(img_tensor)
							except Exception:
								arr = np.asarray(img_tensor)
						pil = Image.fromarray(arr.astype('uint8'), 'RGB')
						name = f'tfds_imagenet_{count + 1}.jpg'
						pil.save(os.path.join(args.images_dir, name))
						count += 1
					print(f"Collected {count} images from tfds into '{args.images_dir}'")
					total_collected = count

	ensure_dir(args.images_dir)
	ensure_dir(args.bins_dir)

	if args.source_dir:
		if not os.path.isdir(args.source_dir):
			print(f"Source dir {args.source_dir} does not exist")
		else:
			print(f"Copying from {args.source_dir} into {args.images_dir} (limit {args.limit})")
			total_collected += copy_from_source(args.source_dir, args.images_dir, args.limit)

	if args.urls_file and total_collected < args.limit:
		if not os.path.isfile(args.urls_file):
			print(f"URLs file {args.urls_file} does not exist")
		else:
			print(f"Downloading URLs from {args.urls_file} into {args.images_dir} (limit {args.limit - total_collected})")
			total_collected += download_from_urls(args.urls_file, args.images_dir, args.limit - total_collected)

	# Fallback: if nothing provided or collected, download one example image (same as previous behavior)
	if total_collected == 0:
		print("No source provided or no images collected. Downloading a single example image.")
		example_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"
		try:
			data = urlopen(example_url, timeout=30).read()
			dest = os.path.join(args.images_dir, 'YellowLabradorLooking_new.jpg')
			with open(dest, 'wb') as out:
				out.write(data)
			total_collected = 1
		except Exception as e:
			print(f"Failed to download example image: {e}")

	print(f"Collected {total_collected} images in '{args.images_dir}'")

	processed = process_images(args.images_dir, args.bins_dir, resize=tuple(args.size))
	print(f"Processed {processed} images into binaries in '{args.bins_dir}'")


if __name__ == '__main__':
	main()