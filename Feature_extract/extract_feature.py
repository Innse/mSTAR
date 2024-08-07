import torch
import os
import time
from dataset.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, './models')
from models import get_custom_transformer, get_model
import argparse
from utils.utils import collate_features, ImgReader
from utils.file_utils import save_hdf5
import openslide
import numpy as np
from multiprocessing import Process
import pandas as pd
import glob
import sys

def save_feature(path, feature):
	s = time.time()
	torch.save(feature, path)
	e = time.time()
	print('Feature is sucessfully saved at: {}, cost: {:.1f} s'.format(path, e-s))


def get_wsi_handle(wsi_path):
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f'{wsi_path} is not found')
    postfix = wsi_path.split('.')[-1]
    if postfix.lower() in ['svs', 'tif', 'ndpi', 'tiff']:
        handle = openslide.OpenSlide(wsi_path)
    elif postfix.lower() in ['jpg', 'jpeg', 'tiff', 'png']:
        handle = ImgReader(wsi_path)
    else:
        raise NotImplementedError(f'{postfix} is not implemented...')
    return handle

def save_hdf5_subprocess(output_path, asset_dict):
	kwargs = {'output_path': output_path, 'asset_dict': asset_dict, 
	   			'attr_dict': None, 'mode': 'w'}
	process = Process(target=save_hdf5, kwargs=kwargs)
	process.start()


def save_feature_subprocess(path, feature):
	kwargs = {'feature': feature, 'path': path}
	process = Process(target=save_feature, kwargs=kwargs)
	process.start()


def light_compute_w_loader(file_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, custom_transformer=None):
	"""
	Do not save features to h5 file to save storage
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, custom_transforms=custom_transformer,
		custom_downsample=custom_downsample, target_patch_size=target_patch_size, fast_read=True)
	kwargs = {'num_workers': 16, 'pin_memory': True} if device.type == "cuda" else {}
	print('Data Loader args:', kwargs)
	loader = DataLoader(dataset=dataset, batch_size=batch_size,  **kwargs, collate_fn=collate_features, prefetch_factor=16)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	features_list = []
	coords_list = []
	_start_time = time.time()
	cal_time = time.time()
	for count, (batch, coords) in enumerate(loader):
		read_time_flag = time.time()
		img_read_time = abs(read_time_flag - cal_time)
		# print('Reading images time:', img_read_time)
		with torch.no_grad():	
			if count % print_every == 0:
				batch_time = time.time()
				print('batch {}/{}, {} files processed, used_time: {} s'.format(
					count, len(loader), count * batch_size, batch_time - _start_time))
			batch = batch.to(device, non_blocking=True)
			features = model(batch)
			features = features.cpu()
			features_list.append(features)
			coords_list.append(coords)
			cal_time = time.time()
		# print('Calculation time: {} s'.format(cal_time-read_time_flag))
		
	features = torch.cat(features_list, dim=0)
	coords = np.concatenate(coords_list, axis=0)
	return features, coords


def find_path_for_tcga_wsi(root, slide_id, ext, datatype):
	if datatype.lower() == 'tcga':
		dirs = os.listdir(root)
		for d in dirs:
			slide_file_path = os.path.join(root, d, slide_id+ext)
			if os.path.exists(slide_file_path):
				return slide_file_path
	else:
		slide_file_path = os.path.join(root, slide_id+ext)
		return slide_file_path


def find_all_wsi_paths(wsi_root, ext):
    """
    find the full wsi path under data_root, return a dict {slide_id: full_path}
    """
    ext = ext[1:]
    result = {}
    all_paths = glob.glob(os.path.join(wsi_root, '**'), recursive=True)
    all_paths = [i for i in all_paths if i.split('.')[-1].lower() == ext.lower()]
    for h in all_paths:
        slide_name = os.path.split(h)[1]
        slide_id = '.'.join(slide_name.split('.')[0:-1])
        result[slide_id] = h
    print("found {} wsi".format(len(result)))
    return result


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--model', type=str)
parser.add_argument('--datatype', type=str)
parser.add_argument('--save_storage', type=str, default='no')


parser.add_argument('--model_checkpoint', type=str, default=None)

args = parser.parse_args()



if __name__ == '__main__':
	process_start_time = time.time()
	print('======================== start running {} ========================'.format(os.path.basename(__file__)))
	print('initializing dataset')

	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError
	csv = pd.read_csv(csv_path)
	csv = csv.applymap(str)
	bags_dataset = Dataset_All_Bags(csv)

	os.makedirs(os.path.join(args.feat_dir, 'h5_files', f'{args.model}'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files', f'{args.model}'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files', f'{args.model}'))

	print('loading model checkpoint:', args.model)

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print('Device:{}, GPU Count:{}'.format(device.type, torch.cuda.device_count()))

	if args.model_checkpoint is not None:
		model = get_model(args.model, device, torch.cuda.device_count(), args.model_checkpoint)
	else:
		model = get_model(args.model, device, torch.cuda.device_count())
		
	custom_transformer = get_custom_transformer(args.model)
	print(custom_transformer)
	total = len(bags_dataset)
	
	# obtain slide_id
	get_slide_id = lambda idx: bags_dataset[idx].split(args.slide_ext)[0]
	# check the exists wsi
	exist_idxs = []
	all_wsi_paths = find_all_wsi_paths(args.data_slide_dir, args.slide_ext)	
	for bag_candidate_idx in range(total):
		slide_id = get_slide_id(bag_candidate_idx)
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		if not os.path.exists(h5_file_path):
			print(h5_file_path, 'does not exist ...')
			continue
		else:
			exist_idxs.append(bag_candidate_idx)

	
	def find_next_file_paths(index, next_num=4):
		# cache next num files
		indexs = [index + i for i in range(next_num)]
		next_ids = [exist_idxs[index % len(exist_idxs)] for index in indexs]
		next_slide_ids = [get_slide_id(i) for i in next_ids]
		file_paths = [all_wsi_paths[nd] for nd in next_slide_ids]
		return file_paths
	

	for index, bag_candidate_idx in enumerate(exist_idxs):
		slide_id = get_slide_id(bag_candidate_idx)
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		if not os.path.exists(h5_file_path):
			print(h5_file_path, 'does not exist ...')
			continue


		slide_file_path = all_wsi_paths[slide_id]

		print('\nprogress: {}/{}, slide_id: {}'.format(bag_candidate_idx, len(exist_idxs), slide_id))

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		next_file_paths = find_next_file_paths(index=index, next_num=5)


		output_h5_path = os.path.join(args.feat_dir, 'h5_files', f'{args.model}', bag_name)
		bag_base, _ = os.path.splitext(bag_name)
		output_feature_path = os.path.join(args.feat_dir, 'pt_files', f'{args.model}', bag_base+'.pt')
		if os.path.isfile(output_feature_path):
			print('skipped {}'.format(output_feature_path))
			continue
		time_start = time.time()

		one_slide_start = time.time()

		wsi = get_wsi_handle(slide_file_path)


		features, coords = light_compute_w_loader(h5_file_path, wsi, 
				model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
				custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size,
				custom_transformer=custom_transformer)
		#save results
		save_feature_subprocess(output_feature_path, features)
		print('coords shape:', coords.shape)
		asset_dict = {'coords': coords}
		save_hdf5_subprocess(output_h5_path, asset_dict=asset_dict)		

		print('time per slide: {:.1f}'.format(time.time() - one_slide_start))
	print('Extracting end!')
	print('Time used for this dataset:{:.1f}'.format(time.time() - process_start_time))
