# ZSIC - Zero Shot Image Classification but more

* Intentionally super simple yet useful.
* Lightweight / faster / $ conscious labelling needs ?
    * Supports CNN based models as vision backbone
* Multilingual labelling needs?
    * Supports Transformers based models as text backbone for multilingual needs
* Supported Vision backbones
    *  ```RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14```
* Supported Languages
    * ```ar, bg, ca, cs, da, de, el, es, et, fa, fi, fr, fr-ca, gl, gu, he, hi, hr, hu, hy, id, it, ja, ka, ko, ku, lt, lv, mk, mn, mr, ms, my, nb, nl, pl, pt, pt, pt-br, ro, ru, sk, sl, sq, sr, sv, th, tr, uk, ur, vi, zh-cn, zh-tw.```
 * Leverages GPU if available, duh!
 * Standing on the shoulder of gaints - OpenAI CLIP, Sentence-Transformers, HuggingFace Transformers,


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
'scores': (0.9940692, 0.0028907193, 0.002512703, 0.0005273586), 
'labels': ('cats', 'lions', 'dogs', 'birds')}
```

##### Spanish

```python
from ZSIC import ZeroShotImageClassification

zsic = ZeroShotImageClassification(lang="es")

preds = zsic(image="http://images.cocodataset.org/val2017/000000039769.jpg",
            candidate_labels=["gatita", "perras", "gatas","leonas"],
            hypothesis_template="una imagen de {}",  # Using a hypothesis_template makes the scores more robust
            )
print(preds)

#Prints the following

{'image': 'http://images.cocodataset.org/val2017/000000039769.jpg', 
'scores': (0.5385471, 0.45578623, 0.003978893, 0.0016878153), 
'labels': ('gatita', 'gatas', 'leonas', 'perras')}
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

### You can use it with over 50 languages
```python
#View Supported lang codes
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
### 💡 Important Tip

* Hypothesis templates default to "A photo of {}" for en but its "{}" for all other lang codes so its on you to pass a nice template for the lang of your choice.
* In the future I will try and Hypothesis templates for all the other languages (hence "lang" is even a parameter)
* Template does make predictions better as mentioned in the origial CLIP paper.
* Quote:
> Another issue we encountered is that it’s relatively rare in our pre-training dataset for the text paired with the image to be just a single word. Usually the > text is a full sentence describing the image in some way. To help bridge this distribution gap, we found that using the prompt template “A photo of a {label}.” > to be a good default that helps specify the text is about the content of the image. This often improves performance over the baseline of using only the label text.  > For instance, just using this prompt improves accuracy on ImageNet by 1.3%.




