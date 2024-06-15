import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torch

from utils import ImageDataset
from data.utils import download_url, check_integrity, noisify

def load_isic(train_path, test_path, imageSize, batch_size, noise_type, noise_rate, logger = None):
    train_df = pd.read_csv(train_path + "ISIC-2017_Training_Part3_GroundTruth.csv")
    test_df=pd.read_csv(test_path + "ISIC-2017_Test_v2_Part3_GroundTruth.csv")

    def add_jpg(x):
        x=x+".jpg"
        return x
    train_df['new_image_id']=train_df['image_id'].apply(add_jpg)
    test_df['new_image_id']=test_df['image_id'].apply(add_jpg)
    def add_label1(x):
        if(x==0.0):
            return 'no'
        else:
            return 'mel'
    def add_label2(x):
        if(x==0.0):
            return 'no'
        else:
            return 'seb'
    def add_label3(x):
        if(x=='nono'):
            return 0
        elif(x=='melno'):
            return 2
        else:
            return 1
    train_df['id']=train_df['melanoma'].apply(add_label1)+train_df['seborrheic_keratosis'].apply(add_label2)
    test_df['id']=test_df['melanoma'].apply(add_label1)+test_df['seborrheic_keratosis'].apply(add_label2)
    train_df['label']=train_df['id'].apply(add_label3)
    test_df['label']=test_df['id'].apply(add_label3)

    df_values = train_df[['label', 'new_image_id']].values
    labels = [x_value[0] for x_value in df_values]
    paths = [train_path + "ISIC-2017_Training_Data/" + x_value[1] for x_value in df_values]
    trainset_array = [(paths[i], pd.to_numeric(labels[i], downcast='float')) for i in range(len(labels))]
    
    # Remove no label
    # train_df=train_df[train_df.label != 2]
    # test_df=test_df[test_df.label != 2]

    transform_train = transforms.Compose(
		[
		# transforms.Resize(128),
		transforms.Resize(imageSize),
		transforms.ToTensor(),
		])
    
    trainset = ImageDataset(trainset_array, transform=transform_train)

	# trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
	#                                           shuffle=False, num_workers=opt.workers)
    # 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											shuffle=False, num_workers=4)

    mean = 0
    std = 0
    for i, data in enumerate(trainloader, 0):
        imgs, labels, index = data
        imgs_mean = np.mean(np.asarray(imgs), axis=(2,3))
        mean += torch.from_numpy(imgs_mean).sum(0)
        imgs_std = np.std(np.asarray(imgs), axis=(2,3))
        std += torch.from_numpy(imgs_std).sum(0)
    mean = mean / len(trainset)
    std = std / len(trainset)

    print("len(trainset)")
    print(len(trainset))

    trainloader = None

    # 

    transform_train = transforms.Compose(
		[
		transforms.Resize(imageSize),
		transforms.RandomCrop(imageSize, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		# transforms.Normalize((mean, mean, mean), (std, std, std))
        transforms.Normalize((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))
		])
    print(transform_train)

    df_values = train_df[['label', 'new_image_id']].values
    labels = np.array([[x_value[0]] for x_value in df_values])
    paths = [train_path + "ISIC-2017_Training_Data/" + x_value[1] for x_value in df_values]

    train_noisy_labels, actual_noise_rate = noisify(dataset="None", train_labels=labels, noise_type=noise_type, noise_rate=noise_rate, random_state=0, nb_classes=3, logger = logger)

    train_noisy_labels=[i[0] for i in train_noisy_labels]
    _train_labels=[i[0] for i in labels]
    noise_or_not = np.transpose(train_noisy_labels)==np.transpose(_train_labels)

    trainset_array = [(paths[i], train_noisy_labels[i]) for i in range(len(labels))]
    trainset = ImageDataset(trainset_array, transform=transform_train)

    transform_test = transforms.Compose(
		[
		transforms.Resize(imageSize),
		transforms.ToTensor(),
		# transforms.Normalize((mean, mean, mean), (std, std, std))
        transforms.Normalize((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))
		])
    print(transform_test)

    df_values = test_df[['label', 'new_image_id']].values
    labels = [x_value[0] for x_value in df_values]
    paths = [test_path + "ISIC-2017_Test_v2_Data/" + x_value[1] for x_value in df_values]
    testset_array = [(paths[i], pd.to_numeric(labels[i], downcast='float')) for i in range(len(labels))]
    testset = ImageDataset(testset_array, transform=transform_test)

    return trainset, testset, noise_or_not, actual_noise_rate