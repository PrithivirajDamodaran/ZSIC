# ZSIC
Zero Shot Image Classification equivalent for HuggingFace Zero Shot Text Classification


## Usage

```python
zsc = ZeroShotImageClassification()
preds = zsc("http://images.cocodataset.org/val2017/000000039769.jpg", ["tv", "cats and remotes", "cats on a pink cloth"])
print(preds)

# prints the following
# {'image': 'http://images.cocodataset.org/val2017/000000039769.jpg', 'scores': [7.725e-05, 1.0, 4.834e-05], 'labels': ['A photo of tv', 'A photo of cats and remotes', 'A photo of cats on a pink cloth']}

```

## You can use CNN or Transformer based pretrained models as vision backbone
```python
#Supported models `RN50`, `RN101`, `RN50x4`, `RN50x16`, `RN50x64`, `ViT-B/32`, `ViT-B/16`, `ViT-L/14`

zsc = ZeroShotImageClassification(model="ViT-B/16")
```

## You can use string templates to make the labels more intuitive
```python
zsc = ZeroShotImageClassification(model="ViT-B/16")
preds = zsc(image="http://images.cocodataset.org/val2017/000000039769.jpg",candidate_labels=["tv", "cats and remotes", "cats on a pink cloth"], hypothesis_template="A image of {}")

# prints the following
# {'image': 'http://images.cocodataset.org/val2017/000000039769.jpg', 'scores': [2.67e-05, 1.0, 7.97e-05], 'labels': ['A image of tv', 'A image of cats and remotes', 'A image of cats on a pink cloth']}
```


