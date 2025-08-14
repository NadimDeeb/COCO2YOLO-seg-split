import splitfolders

splitfolders.ratio(
    input="/mnt/haus/Downloads/cococo",    # folder with sub-folders images/ and labels/
    output="/mnt/haus/Downloads/cocout",   # new root for train/val/test
    seed=42,                           # your chosen seed
    ratio=(.8, .2, .2),                # train/val/test proportions
    group_prefix=None                  # None means split files by name across sub-dirs
)

