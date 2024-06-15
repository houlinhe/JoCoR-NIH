import pandas as pd
import numpy as np
from itertools import chain
import torchvision.transforms as transforms
from utils import ImageDataset
import torch

import os
from glob import glob

def load_nih(train_path, test_path, imageSize, batch_size, logger = None, num_workers = 4):
	data = pd.read_csv(train_path + 'Data_Entry_2017.csv') #.head(1000) # here
	data_image_paths = {os.path.basename(x): x for x in glob(os.path.join('real_data', 'images*', '*', '*.png'))}
	print('Scans found:', len(data_image_paths), ', Total Headers', data.shape[0])

	data['path'] = data['Image Index'].map(data_image_paths.get)

	# TODO: what to do with no label
	# data['Finding Labels'] = data['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
	# print(data['Finding Labels'].str.contains("\|"))
	# print(data['Finding Labels'])

	# ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
	# [ 1.5399,         5.9636,          6.1887,       13.1200,  1.7587,       9.7910,     8.7467,   82.0000,      0.6521,     2.8398,    0.0971,     2.0061,         5.9099,          23.4286,       3.9281]
	# 		+														+												+			  						+

	# Drop No Findings
	data = data.drop(data[data['Finding Labels'] == "No Finding"].index)

	# data = LabelDropper(data, ["Effusion", "Infiltration", "Atelectasis", "Nodule"])
	pd.set_option('display.max_rows', 500)
	print(data)

	print("now dataset length")
	print(len(data))

	all_labels = np.unique(list(chain(*data['Finding Labels'].map(lambda x: x.split('|')).tolist())))
	all_labels = [x for x in all_labels if len(x)>0]
	print(all_labels)
	num_classes = len(all_labels)
	print("num class")
	print(num_classes) # 14
	in_channels = 1
	for c_label in all_labels:
		if len(c_label)>1: # leave out empty labels
			data[c_label] = data['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

	# since the dataset is very unbiased, we can resample it to be a more reasonable collection
	# weight is 0.04 + number of findings
	sample_weights = data['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
	sample_weights /= sample_weights.sum()
	print(len(data))
	# TODO: remember change back
	# data = data.sample(5000, weights=sample_weights) # 20000
	# data = data.sample(1000, weights=sample_weights)
	# print(data.apply(lambda x:print(x)))
	data['disease_vec'] = data.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])

	# all_sample_size = 19000
	# # all_sample_size = 1000
	# num_class_sel = num_classes
	# all_sample_weight = 0.2
	# test_sample_size = round(all_sample_size * all_sample_weight)

	# train_each_sample = round((all_sample_size - test_sample_size) / num_class_sel)
	# test_each_sample = round(test_sample_size / num_class_sel)


	# Indexed - like this
	# ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

	train_df = data

	############
	# Load Test CSV
	############
	test_csv = ""

	test_df = pd.read_csv(test_path + 'Data_Entry_2017.csv') 

	print(train_df)
	print(test_df)

	pd.set_option("display.max_columns", None)
	print(train_df['disease_vec'])


	df_values = train_df[['disease_vec', 'path']].values
	labels = [x_value[0] for x_value in df_values]
	paths = [x_value[1] for x_value in df_values]

	trainset_array = [(paths[i], pd.to_numeric(labels[i], downcast='float')) for i in range(len(labels))]
	# print(trainset_array)
	# print("trainset_array shape")
	# print((torch.Tensor(trainset_array)).shape)

	# transform_train = transforms.Compose(
	# 	[
	# 	# transforms.Resize(128),
	# 	transforms.Resize(224),
	# 	transforms.ToTensor(),
	# 	])

	# trainset = ImageDataset(trainset_array, transform=transform_train)

	# # trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
	# #                                           shuffle=False, num_workers=opt.workers)
	# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
	# 										shuffle=False, num_workers=num_workers)

	# NIH is grayscaled, so the number is this
	mean = 0.495
	std = 0.2288
	#

	transform_train = transforms.Compose(
		[
		transforms.Resize(imageSize),
		transforms.RandomCrop(imageSize, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((mean, mean, mean), (std, std, std))
			# transforms.Normalize((mean[0], mean[1], mean[2]), (0.5, 0.5, 0.5))
		])

	transform_test = transforms.Compose(
		[
		transforms.Resize(imageSize),
		transforms.ToTensor(),
		transforms.Normalize((mean, mean, mean), (std, std, std))
			# transforms.Normalize((mean[0], mean[1], mean[2]), (0.5, 0.5, 0.5))
		])

	logger.info(transform_train)
	logger.info(transform_test)

	trainset = ImageDataset(trainset_array, transform=transform_train)
	# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
	# 										shuffle=True, num_workers=num_workers)

	print("Train Loader Finished")

	df_values = test_df[['disease_vec', 'path']].values
	labels = [x_value[0] for x_value in df_values]
	paths = [x_value[1] for x_value in df_values]

	# trainset = torch.tensor([[read_image(paths[i]), labels[i], i] for i in range(len(labels))])
	testset_array = [(paths[i], pd.to_numeric(labels[i], downcast='float')) for i in range(len(labels))]
	testset = ImageDataset(testset_array, transform=transform_train)

	return trainset, testset