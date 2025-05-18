from datasets import Dataset
from PIL import Image
import os

d={
    "label":["image"],
    "image":[Image.open("Cat_November_2010-1a.jpg")]

}
with open("token.txt","r") as file:
    token=file.readline().strip()
Dataset.from_dict(d).push_to_hub("jlbaker361/testing",token=token)