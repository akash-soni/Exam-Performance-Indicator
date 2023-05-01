# Deployment using elasticbeanstalk

## config for beanstalk

Create a directory .ebextensions inside create file python.config with code : -

```python
option_settings:
  "aws:elasticbeanstalk:container:python":
    WSGIPath: application:application
```

_application_ is the entry point of our application.
Here we will copy app.py and rename it to appliaction.py just for deployement purpose

**Note** also remove _debug=True_ option from application.py

## Data Collection

Dataset Source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
The data consists of 8 column and 1000 rows.
