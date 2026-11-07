"""Final A+tens16 configuration: ten folds and all ten fixed training seeds."""
CONFIG={'dropout':.3,'weight_decay':0.0}
SEEDS=[3003105693, 976400781, 3387213022, 1360466709, 876933081, 2786169488, 207578373, 379423095, 573735355, 1009661775]
N_FOLDS=10
SPLIT_SEED=42


def identity(task):
    if not 0<=task<N_FOLDS*len(SEEDS):
        raise ValueError('Training task must be0..99')
    fold,si=divmod(task,len(SEEDS))
    return fold,SEEDS[si]
