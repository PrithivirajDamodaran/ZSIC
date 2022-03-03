# Zero Shot Image Classification but more


### Usage

```python
from ZSIC import ZeroShotImageClassification

zsic = ZeroShotImageClassification(lang="es")
preds = zsic(image="http://images.cocodataset.org/val2017/000000039769.jpg",
            candidate_labels=["gatita", "perras", "gatas","leonas"],
            hypothesis_template="una imagen de {}",
            )
print(preds)

#Prints the following

{'image': 'http://images.cocodataset.org/val2017/000000039769.jpg', 
'scores': [0.5385471, 0.0016878153, 0.45578623, 0.003978893], 
'labels': ['gatita', 'perras', 'gatas', 'leonas']}
```

### You can use CNN or Transformer based pretrained models as vision backbone
```python
#View Supported models
zsic = ZeroShotImageClassification()
zsic.available_models()

#Prints the following

['RN50',
 'RN101',
 'RN50x4',
 'RN50x16',
 'RN50x64',
 'ViT-B/32',
 'ViT-B/16',
 'ViT-L/14']
```

### You can use it over 50 languages
```python
#View Supported models
zsic = ZeroShotImageClassification()
zsic.available_languages()

#Prints the following

{'ar',
 'bg',
 'ca',
 'cs',
 'da',
 'de',
 'el',
 'en',
 'es',
 'et',
 'fa',
 'fi',
 'fr',
 'fr-ca',
 'gl',
 'gu',
 'he',
 'hi',
 'hr',
 'hu',
 'hy',
 'id',
 'it',
 'ja',
 'ka',
 'ko',
 'ku',
 'lt',
 'lv',
 'mk',
 'mn',
 'mr',
 'ms',
 'my',
 'nb',
 'nl',
 'pl',
 'pt',
 'pt-br',
 'ro',
 'ru',
 'sk',
 'sl',
 'sq',
 'sr',
 'sv',
 'th',
 'tr',
 'uk',
 'ur',
 'vi',
 'zh-cn',
 'zh-tw'}
 ```


### You can use string templates to make the labels more intuitive
```python
zsc = ZeroShotImageClassification(model="ViT-B/16")
preds = zsc(image="http://images.cocodataset.org/val2017/000000039769.jpg",candidate_labels=["tv", "cats and remotes", "cats on a pink cloth"], hypothesis_template="A image of {}")

# prints the following
# {'image': 'http://images.cocodataset.org/val2017/000000039769.jpg', 'scores': [2.67e-05, 1.0, 7.97e-05], 'labels': ['A image of tv', 'A image of cats and remotes', 'A image of cats on a pink cloth']}
```


