from datasets import Dataset
from PIL import Image
import os

d={
    "label":["image"],
    "image":[Image.open("Cat_November_2010-1a.jpg")]

}
token=os.environ["HF_TOKEN"]
Dataset.from_dict(d).push_to_hub("jlbaker361/testing",token=token)