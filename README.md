# Zero Shot Image Classification but more

### Installation
```python
!pip install git+https://github.com/PrithivirajDamodaran/ZSIC.git
```

### Usage

##### English
```python
from ZSIC import ZeroShotImageClassification

zsic = ZeroShotImageClassification()


#Predictions
preds = zsic(image="http://images.cocodataset.org/val2017/000000039769.jpg",
            candidate_labels=["birds", "lions", "cats","dogs"], 
            )
print(preds)

#Prints the following

{'image': 'http://images.cocodataset.org/val2017/000000039769.jpg', 
'scores': [0.00046659182, 0.0024660423, 0.9949238, 0.002143612], 
'labels': ['birds', 'lions', 'cats', 'dogs']}
```

##### Spanish

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
zsic = ZeroShotImageClassification(model="RN50")
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


