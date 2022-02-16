from typing import List, Union
import torch
import clip
import PIL
from PIL import Image
import requests
import os


class ZeroShotImageClassification():

  def __init__(self, 
               *args, 
               **kwargs,):
    
         device = "cuda" if torch.cuda.is_available() else "cpu" 
         if "model" in kwargs:
            model = kwargs["model"] 
            self.model, self.preprocess = clip.load(model, device=device)
         else:
            model = "ViT-B/32"    
            self.model, self.preprocess = clip.load(model, device=device)



  def _load_image(self, image: str) -> "PIL.Image.Image":
      """
      Loads `image` to a PIL Image.
      Args:
          image (`str` ):
              The image to convert to the PIL Image format.
      Returns:
          `PIL.Image.Image`: A PIL Image.
      """
      if isinstance(image, str):
          if image.startswith("http://") or image.startswith("https://"):
              image = PIL.Image.open(requests.get(image, stream=True).raw)
          elif os.path.isfile(image):
              image = PIL.Image.open(image)
          else:
              raise ValueError(
                  f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
              )
      elif isinstance(image, PIL.Image.Image):
          image = image
      else:
          raise ValueError(
              "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
          )
      image = PIL.ImageOps.exif_transpose(image)
      image = image.convert("RGB")
      return image            

  def __call__(
        self, 
        image: Union[str, List[str]],
        candidate_labels: Union[str, List[str]],
        *args,
        **kwargs,
    ):

        """
        Classify the image using the candidate labels given
        Args:
            image (`str`):
                Fully Qualified path of a local image or URL of image
            candidate_labels (`str` or `List[str]`):
                The set of possible class labels to classify each sequence into. Can be a single label, a string of
                comma-separated labels, or a list of labels.
            hypothesis_template (`str`, *optional*, defaults to `"A photo of {}."`):
                The template used to turn each label into a string. This template must include a {} or
                similar syntax for the candidate label to be inserted into the template. 
           model (`str`, *optional*, defaults to `ViT-B/32`):
                Any one of the CNN or Transformer based pretrained models can be used as Vision backbone. 
                `RN50`, `RN101`, `RN50x4`, `RN50x16`, `RN50x64`, `ViT-B/32`, `ViT-B/16`, `ViT-L/14`
           top_k (`int`, *optional*, defaults to 5):
                The number of top labels that will be returned by the pipeline. If the provided number is higher than
                the number of labels available in the model configuration, it will default to the number of labels.
        Return:
            A `dict` or a list of `dict`: Each result comes as a dictionary with the following keys:
            - **image** (`str`) -- The image for which this is the output.
            - **labels** (`List[str]`) -- The labels sorted by order of likelihood.
            - **scores** (`List[float]`) -- The probabilities for each of the labels.
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if "hypothesis_template" in kwargs:
            hypothesis_template = kwargs["hypothesis_template"] 
        else:
            hypothesis_template = "A photo of {}"

        if isinstance(candidate_labels, str):
            candidate_labels = [hypothesis_template.format(candidate_label) for candidate_label in candidate_labels.split(",")]
        else:    
            candidate_labels = [hypothesis_template.format(candidate_label) for candidate_label in candidate_labels]

        if  "top_k" in kwargs:
             top_k = kwargs["top_k"] 
        else:
             top_k = len(candidate_labels)
        
        img = self.preprocess(self._load_image(image)).unsqueeze(0).to(device)
        text = clip.tokenize(candidate_labels).to(device)

        with torch.no_grad():
            image_features = self.model.encode_image(img)
            text_features = self.model.encode_text(text)
            
            logits_per_image, logits_per_text = self.model(img, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            scores = probs.flatten()

        preds = {}
        preds["image"] = image
        preds["scores"] = list(scores)
        preds["labels"] = candidate_labels
        return preds
