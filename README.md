# CRUMPL: (insert acronym definition here)

Stanford CS231n final project, Spring 2022

Authors (alphabetical by last name):
* Maximilian Du
* Niveditha Iyer
* Tejas Narayanan

## Installing dataset

All data should be installed under the `data` subdirectory.

We are using the validation subset of ImageNet as our full dataset, since
the full ImageNet dataset is extremely large.

To access ImageNet data, sign up for an account at
[https://image-net.org/](https://image-net.org/). Then, download
the "blurred validation images" under the "Face obfuscation in ILSVRC"
heading. This will download `val_blurred.zip`. Extract this archive into
the `data/val_blurred` directory. The structure should look like the
following:

```
CRUMPL
│  ...
└─ data
   └─ val_blurred
      └─ n01440764
      └─ n01443537
      └─ ...
```

## Generating data

Download Blender from
[https://www.blender.org/download/](https://www.blender.org/download/).
Then, open `paper_gen.blend` and navigate to the Scripting tab at the
top. In the console that appears on the bottom left, enter the following
two lines to install `tqdm`, a progress bar library:

```python
>>> import pip
>>> pip.main(['install', 'tqdm'])
```

Replace the constants `DATA_PATH` and `EXPORT_PATH` based
on your computer's file path. Finally, run the file by pressing the run
button on the menu bar of the code editor area.