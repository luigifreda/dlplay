# ******************************************************************************
# This file is part of dlplay
# 
# Copyright (C) Luigi Freda <luigi dot freda at gmail dot com>
# 
# dlplay is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# dlplay is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with dlplay. If not, see <http://www.gnu.org/licenses/>.
# ******************************************************************************
from enum import Enum


import sklearn


class ProblemType(Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTI_CLASS_CLASSIFICATION = "multi_class_classification"


class ToyDatasetType(Enum):
    IRIS = "iris"
    DIABETES = "diabetes"
    DIGITS = "digits"
    LINNERUD = "linnerud"
    WINE = "wine"
    BREAST_CANCER = "breast_cancer"
    GENERIC_CLASSIFICATION = "generic_classification"


# Toy datasets from scikit-learn
# from https://scikit-learn.org/stable/datasets/toy_dataset.html


## Logistic Regression - Classification


# multi-class classification
def load_iris_data(test_size=0.25, random_state=42):
    iris = sklearn.datasets.load_iris()
    X = iris.data
    y = iris.target
    return sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


# multi-class classification
def load_digits_data(test_size=0.25, random_state=42):
    digits = sklearn.datasets.load_digits()
    X = digits.data
    y = digits.target
    return sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


#  Load generic classification data (can be used for binary or multi-class classification)
def load_generic_classification_data(
    n_samples=500,
    n_features=2,
    n_classes=2,
    n_informative=2,
    n_redundant=0,
    test_size=0.25,
    random_state=42,
):
    X, y = sklearn.datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=random_state,
    )
    X = sklearn.preprocessing.StandardScaler().fit_transform(X)
    # train test split
    return sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


# binary classification
def load_breast_cancer_data(test_size=0.25, random_state=42):
    breast_cancer = sklearn.datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    return sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


# multi-class classification
def load_wine_data(test_size=0.25, random_state=42):
    wine = sklearn.datasets.load_wine()
    X = wine.data
    y = wine.target
    return sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


## Regression


# regression
def load_diabetes_data(test_size=0.25, random_state=42):
    diabetes = sklearn.datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    return sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


# multi-task regression
def load_linnerud_data(test_size=0.25, random_state=42):
    linnerud = sklearn.datasets.load_linnerud()
    X = linnerud.data
    y = linnerud.target
    return sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


class SklearnToyDatasetLoader:
    def __init__(
        self,
        dataset_name: ToyDatasetType,
        problem_type: ProblemType,
        test_size: float = 0.25,
        random_state: int = 42,
        use_dataset_scaler: bool = False,
        params: dict = None,  # params for generic classification dataset
    ):
        self.dataset_name = dataset_name
        self.problem_type = problem_type
        self.test_size = test_size
        self.random_state = random_state
        self.params = params
        self.use_dataset_scaler = use_dataset_scaler
        self.scaler = (
            sklearn.preprocessing.StandardScaler() if use_dataset_scaler else None
        )
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.load_data()

    def get_dataset_name(self):
        return self.dataset_name.value

    def get_problem_type(self):
        return self.problem_type.value

    def load_data(self):
        if self.dataset_name == ToyDatasetType.IRIS:
            if self.problem_type != ProblemType.MULTI_CLASS_CLASSIFICATION:
                raise ValueError(
                    f"Dataset {self.dataset_name} is a multi-class classification dataset"
                )
            self.X_train, self.X_test, self.y_train, self.y_test = load_iris_data(
                test_size=self.test_size, random_state=self.random_state
            )
        elif self.dataset_name == ToyDatasetType.DIABETES:
            if self.problem_type != ProblemType.REGRESSION:
                raise ValueError(f"Dataset {self.dataset_name} is a regression dataset")
            self.X_train, self.X_test, self.y_train, self.y_test = load_diabetes_data(
                test_size=self.test_size, random_state=self.random_state
            )
        elif self.dataset_name == ToyDatasetType.DIGITS:
            if self.problem_type != ProblemType.MULTI_CLASS_CLASSIFICATION:
                raise ValueError(
                    f"Dataset {self.dataset_name} is a multi-class classification dataset"
                )
            self.X_train, self.X_test, self.y_train, self.y_test = load_digits_data(
                test_size=self.test_size, random_state=self.random_state
            )
        elif self.dataset_name == ToyDatasetType.LINNERUD:
            if self.problem_type != ProblemType.REGRESSION:
                raise ValueError(f"Dataset {self.dataset_name} is a regression dataset")
            self.X_train, self.X_test, self.y_train, self.y_test = load_linnerud_data(
                test_size=self.test_size, random_state=self.random_state
            )
        elif self.dataset_name == ToyDatasetType.WINE:
            if self.problem_type != ProblemType.MULTI_CLASS_CLASSIFICATION:
                raise ValueError(
                    f"Dataset {self.dataset_name} is a multi-class classification dataset"
                )
            self.X_train, self.X_test, self.y_train, self.y_test = load_wine_data(
                test_size=self.test_size, random_state=self.random_state
            )
        elif self.dataset_name == ToyDatasetType.BREAST_CANCER:
            if self.problem_type != ProblemType.BINARY_CLASSIFICATION:
                raise ValueError(
                    f"Dataset {self.dataset_name} is a binary classification dataset"
                )
            self.X_train, self.X_test, self.y_train, self.y_test = (
                load_breast_cancer_data(
                    test_size=self.test_size, random_state=self.random_state
                )
            )
        elif self.dataset_name == ToyDatasetType.GENERIC_CLASSIFICATION:
            if self.params is not None:
                if "n_classes" not in self.params:
                    raise ValueError(
                        "n_classes is required for generic classification dataset"
                    )
                if "n_features" not in self.params:
                    raise ValueError(
                        "n_features is required for generic classification dataset"
                    )
            self.X_train, self.X_test, self.y_train, self.y_test = (
                load_generic_classification_data(
                    test_size=self.test_size,
                    random_state=self.random_state,
                    **self.params,
                )
            )
        else:
            raise ValueError(f"Dataset {self.dataset_name} not found")
        if self.scaler is not None:
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
