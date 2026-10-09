"""Dataset constants and the poison/target cases published in the paper.

Normalization constants: the paper states images are normalized but does not list the values, so
these come from the reference implementation (consts.py) -- see DESIGN_DECISIONS.md section 5.

The case tables are verbatim from the paper, Table 4 (CIFAR-10) and Table 5 (ImageNet). Using them
instead of regenerating cases with our own RNG makes the experiments reproduce the published
settings exactly rather than merely the published methodology.
"""

CIFAR10_MEAN = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
CIFAR10_STD = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Paper Table 4: target class, poison class, target ID (index into the CIFAR-10 validation set in
# the canonical order of the dataset as downloaded from cs.toronto.edu), random seed.
# Note the paper's Table 4 caption says "for ImageNet experiments" but the target IDs are relative
# to the CIFAR-10 validation dataset; the seed block 2*** is the CIFAR-10 evaluation block per
# Appendix B, so these are the CIFAR-10 cases.
CIFAR10_CASES = (
    ("dog", "frog", 8745, 2000000000),
    ("frog", "truck", 1565, 2100000000),
    ("frog", "bird", 2138, 2110000000),
    ("airplane", "dog", 5036, 2111000000),
    ("airplane", "ship", 1183, 2111100000),
    ("cat", "airplane", 7352, 2111110000),
    ("automobile", "frog", 3544, 2111111000),
    ("truck", "cat", 3676, 2111111100),
    ("automobile", "ship", 9882, 2111111110),
    ("automobile", "cat", 3028, 2111111111),
)

# Paper Table 5: same structure, target ID relative to the ILSVRC2012 validation dataset with
# syn-sets ordered by increasing numerical value (the torchvision default).
IMAGENET_CASES = (
    ("otter", "Labrador retriever", 18047, 1000000000),
    ("warthog", "bib", 17181, 1100000000),
    ("orange", "radiator", 37530, 1110000000),
    ("theater curtain", "maillot", 42720, 1111000000),
    ("hartebeest", "capuchin", 17580, 1111100000),
    ("burrito", "plunger", 48273, 1111110000),
    ("jackfruit", "spider web", 47776, 1111111000),
    ("king snake", "hyena", 2810, 1111111100),
    ("flat-coated retriever", "alp", 10281, 1111111110),
    ("window screen", "hard disc", 45236, 1111111111),
)
