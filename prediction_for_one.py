import torch
import torchvision
import argparse

import model_builder,data_setup

# Creating a parser
parser = argparse.ArgumentParser()



# Get a model path
parser.add_argument("--model_path",
                    default="/home/sujata/Downloads/karan_data/ViT_base/models/100_epoch_1layer_lre4_ViT_base_model.pth",
                    type=str,
                    help="target model to use for prediction filepath")

args = parser.parse_args()

# Setup hyperparameters(just change these parameter according to the model we are using)
D_MODEL= 768
IMAGE_SIZE=(224,224)
PATCH_SIZE=(16,16)
N_CHANNELS=3
N_HEADS=12
N_LAYERS=1
LAYER_MLP=4*D_MODEL # Neuron in each layer of mlp

# Setup class names
class_names =["AB","BP","HL","YMV"]


# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get the image path
IMG_PATH = "/home/sujata/Downloads/karan_data/ViT_base/SoyMulti/test/HL/hlimg(67).jpg"

print(f"[INFO] Predicting on {IMG_PATH}")

# Function to load in the model
def load_model(filepath=args.model_path):
  # Need to use same hyperparameters as saved model 
  model=model_builder.VisionTransformer(
    d_model=D_MODEL,
    n_classes=len(class_names),
    img_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    n_channels=N_CHANNELS,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    layer_mlp=LAYER_MLP
  ).to(device)


  print(f"[INFO] Loading in model from: {filepath}")
  # Load in the saved model state dictionary from file                               
  model.load_state_dict(torch.load(filepath))

  return model

# Function to load in model + predict on select image
def predict_on_image(image_path=IMG_PATH, filepath=args.model_path):
  # Load the model
  model = load_model(filepath)

  # Load in the image and turn it into torch.float32 (same type as model)
  image = torchvision.io.read_image(str(IMG_PATH)).type(torch.float32)

  # Preprocess the image to get it between 0 and 1
  image = image / 255.

  # Resize the image to be the same size as the model
  transform = torchvision.transforms.Resize(size=(224, 224))
  image = transform(image) 

  # Predict on image
  model.eval()
  with torch.inference_mode():
    # Put image to target device
    image = image.to(device)

    # Get pred logits
    pred_logits = model(image.unsqueeze(dim=0)) # make sure image has batch dimension (shape: [batch_size, height, width, color_channels])

    # Get pred probs
    pred_prob = torch.softmax(pred_logits, dim=1)

    # Get pred labels
    pred_label = torch.argmax(pred_prob, dim=1)
    pred_label_class = class_names[pred_label]

  print(f"[INFO] Pred class: {pred_label_class}, Pred prob: {pred_prob.max():.3f}")

if __name__ == "__main__":
  predict_on_image()