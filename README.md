## Boston Housing Prediction

In this project I used the Boston Housing Dataset (http://lib.stat.cmu.edu/datasets/boston) to build a little web app than allow to make predictions give some inputs.

I select as features the size of the house (zone), the number of rooms and the distance to the downtown.

The model is a SVM using as kernel the rbf.

## Technologies

- Python

- Pandas

- Numpy

- Sklearn

- Flask

- Docker

  

## How to use

1. Clone the repository `git clone https://github.com/diego6662/boston_house.git`

2. Install the requirements `pip install -r requirements.txt`

3. Run the app `python boston_housing/app.py`

   

or simply you can use the Dockerfile to build the image and run the container.