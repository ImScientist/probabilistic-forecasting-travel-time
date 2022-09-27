import pandas as pd
from abc import abstractmethod


class MyModel:
    """ Abstract class that holds all model relevant settings """

    def __init__(self):
        self.model = None

    @abstractmethod
    def df_to_dataset(
            self,
            df: pd.DataFrame,
            shuffle_buffer_size: int = 0,
            batch_size: int = None,
            prefetch_size: int = None,
            cache: bool = False,
            **kwargs
    ):
        """ Create a dataset from a dataframe """

    @abstractmethod
    def init_model(self, ds, **kwargs):
        """ Initialize a model """

    @abstractmethod
    def evaluate_model(self, ds, log_dir: str):
        """ Evaluate model """

    @abstractmethod
    def load(self, load_dir: str):
        """ Load the model and all relevant class attributes """

    @abstractmethod
    def save(self, load_dir: str):
        """ Save the model and other class attributes """
