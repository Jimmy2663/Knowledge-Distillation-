"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine,KD_engine, model_builder, utils
from pathlib import Path
from torchvision import models
from torchvision import transforms
from timeit import default_timer as timer
from torchvision.transforms import TrivialAugmentWide

#Setup saving directory
SAVE_DIR = Path("Save_dir_res18")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# Setup hyperparameters
NUM_EPOCHS =50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
N_CHANNELS=3

# Setup directories
train_dir = "/home/sgudge/Dataset/PlantVillageSplit_data/train"
test_dir = "/home/sgudge/Dataset/PlantVillageSplit_data/val"              #Note: this is the validation directory
real_test_dir = "/home/sgudge/Dataset/PlantVillageSplit_data/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((224, 224)),
  TrivialAugmentWide(num_magnitude_bins=31),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, real_test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    real_test_dir=real_test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Starting timer for complete experiment 
start=timer()

#-----------------------------------------------------------------------------------------------------------------------------------

print(f"\n********************************Eperimentation_for_Resnet-18_model_begins_here************************************************")

# Create model with help from model_builder.py
student1=model_builder.ResNet18(num_classes=len(class_names))

# Set loss and optimizer
loss_fn2 = torch.nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(student1.parameters(),
                             lr=LEARNING_RATE)

# Start the timer 
start_time=timer()

# Start training with help from engine.py
engine.train_and_test(model=student1,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             real_test_dataloader=real_test_dataloader,
             loss_fn=loss_fn2,
             optimizer=optimizer2,
             epochs=NUM_EPOCHS,
             device=device,
             class_names=class_names,
             save_dir=SAVE_DIR
             )

# End the timer
end_time=timer()
experimentation_time=end_time-start_time

print(f"\nTotal Experimentation time for Model : {utils.format_time(experimentation_time)}")

#saving model summary using save_model_summary.py

utils.save_model_summary(student1,input_size=[32,3,224,224], file_path=SAVE_DIR/"model_summary.txt", use_torchinfo=True, device=device)

print(f"\nModel_summary saved to {SAVE_DIR}/model_summary.txt ")
print(f"\n--------------------------------------------------------------------------------------------------------------------")

# Save the model with help from utils.py
utils.save_model(model=student1,
                 target_dir=SAVE_DIR,
                 model_name="End_model.pth")


print(f"\n----------------------------------------------------------------------------------------------------------------------")
print(f"\nExperiment completed for Model: Student1 !!! ")
print(f"\n----------------------------------------------------------------------------------------------------------------------")


