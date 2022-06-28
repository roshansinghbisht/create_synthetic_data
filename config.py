# path to your own data and coco file
train_data_dir = "created_data/output_2022_06_28 11_45_30.006451/train/images"
train_coco = "created_data/output_2022_06_28 11_45_30.006451/train/output.json"
test_data_dir = "created_data/output_2022_06_28 11_45_30.006451/test/images"

# Batch size
train_batch_size = 5

# Params for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Params for training

# Two classes; Only target class or background
num_classes = 36
num_epochs = 5

lr = 0.005
momentum = 0.9
weight_decay = 0.005
