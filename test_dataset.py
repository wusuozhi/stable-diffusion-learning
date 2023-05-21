from ldm.data.simple import hf_dataset
import pandas as pd
from datasets import Dataset
from PIL import Image
from torchvision import transforms
from einops import rearrange


if __name__ == "__main__":
    # ds = hf_dataset('lambdalabs/pokemon-blip-captions')
    # print(ds)

    def preprocess(example):
        # img_path = example['image'][0]
        # print(img_path)[0]
        # txt = example['txt']
        trans = transforms.Compose(
            [
                transforms.Resize(512,interpolation=3),
                transforms.RandomCrop(size=512),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
            ]
        )
        # img = Image.open(img_path)
        # img_tensor = trans(img)
        processed = {}
        processed['image'] = [trans(Image.open(img_path).convert('RGB')) for img_path in example['image']]
        processed['txt'] = example['txt']
        return processed

    df = pd.read_csv('data/pokemon_data/train.csv')
    df_dict = {'image':df['img_path'].tolist(),'txt':df['text'].tolist()}
    ds = Dataset.from_dict(df_dict)
    ds.set_transform(preprocess)
    print(ds[0]['image'].shape)
