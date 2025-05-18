from datasets import Dataset
from PIL import Image

d={
    "label":["image"],
    "image":[Image.open("Cat_November_2010-1a.jpg")]

}

Dataset.from_dict(d).push_to_hub("jlbaker361/testing")