
# !pip install ludwig --user

import pandas as pd
from ludwig.api import LudwigModel

df = pd.read_csv('ludwig/rotten_tomatoes.csv')

model = LudwigModel(config='ludwig/rotten_tomatoes.yaml')
results = model.train(dataset=df)
# Lock 1420789236640 acquired on C:\swc\.lock_preprocessing
# Lock 1420789236640 released on C:\swc\.lock_preprocessing

model = LudwigModel.load('results/experiment_run/model')

predictions, _ = model.predict(dataset='ludwig/rotten_tomatoes_test.csv')
predictions.head()

