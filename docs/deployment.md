# Deployment using elasticbeanstalk

## STEP 1 - config for beanstalk

Create a directory .ebextensions inside create file python.config with code : -

```python
option_settings:
  "aws:elasticbeanstalk:container:python":
    WSGIPath: application:application
```

_application_ is the entry point of our application.
Here we will rename it to appliaction.py just for deployement purpose

**Note** also remove _debug=True_ option from application.py

## STEP 2 - AWS setup

1. Login to AWS account
2. Search for Elasic Beanstalk Service
3. Elasic Beanstalk reqires configuration and our code is in github Repository.
   We need to create a Pipeline so that code can go from repository to beanstalk even on any change
   In order to create this we create **AWS codepipeline**, this pipeline is called **Continuous Delivery** pipeline.

![DataSet directory](./img/deployment_process.jpg?raw=true "Dataset directory")

4. Create EBS Application "exam-performance"
5. Use AWS Codepipeline service to create a CD Pipeline
