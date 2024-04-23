# Inventories randomly selected for evaluation

Range: 1060 - 8000

Python code:

```python
import random

lower = 1060
upper = 8000

for _ in range(10):
   print(random.randint(lower, upper))

5457
7710
3351
5544
5075
1255
5435
4439
5726
4023
```

For predicting (Google format) with a model file `model.pt`:

```console
# Replace MODEL_DIR
MODEL_DIR=desert-lake-284

poetry run python scripts/predict_inventories.py --inventory 5457 7710 3351 5544 5075 1255 5435 4439 5726 4023 --format google -o ${MODEL_DIR}/test_inventories.csv --model ${MODEL_DIR}/model.pt
```

For GPU use, specify device, e.g.:

```console
CUDA_VISIBLE_DEVICES=MIG-6f57cdc3-7a4b-5db9-a749-b91d7473f802 poetry run python scripts/predict_inventories.py --inventory 5457 7710 3351 5544 5075 1255 5435 4439 5726 4023 --format google -o ${MODEL_DIR}/test_inventories.csv --model ${MODEL_DIR}/model.pt --device cuda:0
````
