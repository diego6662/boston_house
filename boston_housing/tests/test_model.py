from boston_housing.model import get_dataset, build_model
import pytest
import numpy as np


def test_dataset_shape():
    df = get_dataset()
    assert df.shape == (506,14), 'The dataset shape doesnt match'

def test_dataset_columns():
    df = get_dataset()
    assert all(df.columns.values == [
        'crim','zn','indus', 'chas', 'nox', 'rm', 'age', 
        'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv'
    ]), 'The columns values are different'

def test_model():
    test_data = np.array([[450000 / 25000, 6.5750, 4.0900]])
    model = build_model(training=False)
    predict = model.predict(test_data)[0]
    assert predict == pytest.approx(26.891347311852385)