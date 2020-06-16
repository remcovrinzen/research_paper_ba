from eda import Eda
from cleaning import Cleaning
from models import Models
from features import Features
import warnings


def main():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import sklearn

    # eda = Eda()
    # cleaned_data = Cleaning()
    # features = Features()
    models = Models()


if __name__ == "__main__":
    main()
