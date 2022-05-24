import random, os, torch


#Import ViT Packages
from models.imagetransformer import ImageTransformer
from models.DDFA import *
from utils_ViT import load_pretrained_weights, PRETRAINED_MODELS, as_tuple, resize_positional_embedding_
from transformer import *
from dataset_utils.training_dataset_creation import VideoTrainDataset


# Import 3DDFA Packages
import yaml
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA

seed = 17
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

paths = []
train_dir_real = 'train_path_to_real_set'
train_dir_fake = 'train_path_to_faceswap'
train_dir_fake_2 = 'train_path_to_face2face'
train_dir_fake_3 = 'train_path_to_deepfakes'
train_dir_fake_4 = 'train_path_to_neuraltextures'

valid_dir_real   = 'validation_path_to_real_set'
valid_dir_fake   = 'validation_path_to_fake_faceswap'
valid_dir_fake_2 = 'validation_path_to_fake_face2face'
valid_dir_fake_3 = 'validation_path_to_fake_deepfakes'
valid_dir_fake_4 = 'validation_path_to_fake_neuraltextures'

paths.append(train_dir_real)
paths.append(train_dir_fake)
paths.append(train_dir_fake_2)
paths.append(train_dir_fake_3)
paths.append(train_dir_fake_4)
paths.append(valid_dir_real)
paths.append(valid_dir_fake)
paths.append(valid_dir_fake_2)
paths.append(valid_dir_fake_3)
paths.append(valid_dir_fake_4)

batch_size = 6
train_loader, valid_loader = VideoTrainDataset.get_image_batches(paths, batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ImageTransformer('B_16_imagenet1k', pretrained=True, image_size = 384, num_classes = 2,
                        seq_embed=True, hybrid=True, variant='image', device=device)

epochs = 15
lr = 3e-3
# gamma = 0.7

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in (train_loader):
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in (valid_loader):
            data = data.to(device)
            label = label.to(device)
            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)    
    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

