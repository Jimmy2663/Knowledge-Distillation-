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
SAVE_DIR = Path("Save_dir_T2_A50_pv")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories and store their paths
TEACHER_DIR = SAVE_DIR / "teacher"
STUDENT1_DIR = SAVE_DIR / "student1"
STUDENT2_DIR = SAVE_DIR / "student2"

# Make sure subdirectories exist
TEACHER_DIR.mkdir(parents=True, exist_ok=True)
STUDENT1_DIR.mkdir(parents=True, exist_ok=True)
STUDENT2_DIR.mkdir(parents=True, exist_ok=True)

# Setup hyperparameters
TEACHER_EPOCHS =100
STUDENT_EPOCHS =30
BATCH_SIZE = 32
STUDNENT_LR = 0.0001
TEACHER_LR=0.0001
N_CHANNELS=3
T=2
ALPHA=0.50

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
# Train Teacher model VGG16 

print(f"\n********************************Eperimentation_for_Teacher_model_begins_here************************************************")


# Create model with help from model_builder.py
Teacher=model_builder.VGG16(
    input_shape=N_CHANNELS,
    output_shape=len(class_names),
    dropout=0.1
).to(device)

# Set loss and optimizer
loss_fn1 = torch.nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(Teacher.parameters(),
                             lr=TEACHER_LR)

# Start the timer 
start_time=timer()

# Start training with help from engine.py
engine.train_and_test(model=Teacher,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             real_test_dataloader=real_test_dataloader,
             loss_fn=loss_fn1,
             optimizer=optimizer1,
             epochs=TEACHER_EPOCHS,
             device=device,
             class_names=class_names,
             save_dir=TEACHER_DIR
             )

# End the timer
end_time=timer()
experimentation_time=end_time-start_time

print(f"\nTotal Experimentation time for Teacher : {utils.format_time(experimentation_time)}")

#saving model summary using save_model_summary.py

utils.save_model_summary(Teacher,input_size=[32,3,224,224], file_path=TEACHER_DIR/"model_summary_teacher.txt", use_torchinfo=True, device=device)

print(f"\nModel_summary saved to {TEACHER_DIR}/model_summary_teacher.txt ")
print(f"\n--------------------------------------------------------------------------------------------------------------------")

# Save the model with help from utils.py
utils.save_model(model=Teacher,
                 target_dir=TEACHER_DIR,
                 model_name="Teacher_model.pth")


print(f"\n----------------------------------------------------------------------------------------------------------------------")
print(f"\nExperiment completed for Model: Teacher VGG16 !!! ")
print(f"\n----------------------------------------------------------------------------------------------------------------------")



#------------------------------------------------------------------------------------------------------------------------------------
# STUDENT1 WITHOUT KD

print(f"\n********************************Eperimentation_for_Student_1(without_KD)_model_begins_here************************************************")

# Create model with help from model_builder.py
student1=model_builder.ResNet18(num_classes=len(class_names))

# Set loss and optimizer
loss_fn2 = torch.nn.CrossEntropyLoss()
optimizer2 = torch.optim.Adam(student1.parameters(),
                             lr=STUDNENT_LR)

# Start the timer 
start_time=timer()

# Start training with help from engine.py
engine.train_and_test(model=student1,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             real_test_dataloader=real_test_dataloader,
             loss_fn=loss_fn2,
             optimizer=optimizer2,
             epochs=STUDENT_EPOCHS,
             device=device,
             class_names=class_names,
             save_dir=STUDENT1_DIR
             )

# End the timer
end_time=timer()
experimentation_time=end_time-start_time

print(f"\nTotal Experimentation time for Student1 : {utils.format_time(experimentation_time)}")

#saving model summary using save_model_summary.py

utils.save_model_summary(student1,input_size=[32,3,224,224], file_path=STUDENT1_DIR/"model_summary_student1.txt", use_torchinfo=True, device=device)

print(f"\nModel_summary saved to {STUDENT1_DIR}/model_summary_student1.txt ")
print(f"\n--------------------------------------------------------------------------------------------------------------------")

# Save the model with help from utils.py
utils.save_model(model=student1,
                 target_dir=STUDENT1_DIR,
                 model_name="student1_model.pth")


print(f"\n----------------------------------------------------------------------------------------------------------------------")
print(f"\nExperiment completed for Model: Student1 !!! ")
print(f"\n----------------------------------------------------------------------------------------------------------------------")


#--------------------------------------------------------------------------------------------------------------------------------
# STUDENT 2 WITH CE + KD 

print(f"\n********************************Eperimentation_for_Student2_model_(with CE+KD)_begins_here************************************************")

# Create model with help from model_builder.py
student2=model_builder.ResNet18(num_classes=len(class_names))

# Set loss and optimizer
loss_fn3 = torch.nn.CrossEntropyLoss()
optimizer3 = torch.optim.Adam(student2.parameters(),
                             lr=STUDNENT_LR)

# Start the timer 
start_time=timer()

# Start training with help from KD_engine.py
KD_engine.train_and_test_with_KD(teacher=Teacher,
                                 student=student2,
                                 train_dataloader=train_dataloader,
                                 test_dataloader=test_dataloader,
                                 real_test_dataloader=real_test_dataloader,
                                 optimizer=optimizer3,
                                 loss_fn=loss_fn3,
                                 epochs=STUDENT_EPOCHS,
                                 T=T,
                                 soft_target_loss_weight=1-ALPHA,
                                 ce_loss_weight=ALPHA,
                                 device=device,
                                 class_names=class_names,
                                 save_dir=STUDENT2_DIR
                                 )

# End the timer
end_time=timer()
experimentation_time=end_time-start_time

print(f"\nTotal Experimentation time for Student2 : {utils.format_time(experimentation_time)}")

#saving model summary using save_model_summary.py

utils.save_model_summary(student2,input_size=[32,3,224,224], file_path=STUDENT2_DIR/"model_summary_student2.txt", use_torchinfo=True, device=device)

print(f"\nModel_summary saved to {STUDENT2_DIR}/model_summary_student2.txt ")
print(f"\n--------------------------------------------------------------------------------------------------------------------")

# Save the model with help from utils.py
utils.save_model(model=student2,
                 target_dir=STUDENT2_DIR,
                 model_name="student2_model.pth")


print(f"\n----------------------------------------------------------------------------------------------------------------------")
print(f"\nExperiment completed for Model: Student2 !!! ")
print(f"\n----------------------------------------------------------------------------------------------------------------------")

end=timer()

experimentation_time=end-start
print(f"\nComplete time required for the experimentation is : {utils.format_time(experimentation_time)}")
print(f"\n----------------------------------------------------------------------------------------------------------------------")