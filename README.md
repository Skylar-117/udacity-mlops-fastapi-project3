# Udacity MLOps - Deploy FastAPI on Heroku

We will use AWS S3 to save the data and model files. Use DVC for data/model version control and link it to remote S3 bucket. Set up CI(continuous integration) workflow via Github Actions. Deploy machine learning pipeline via FastAPI on Heroku. Enable CD(continuous deployment) workflow on Heroku.


## Environment Setup
```shell
# Create a directory for the project and clone the project from Github
mkdir udacity && cd udacity
git clone https://github.com/Skylar-117/nd0821-c3-starter-code.git
cd nd0821-c3-starter-code

# Create a new conda environment and install necessary Python libraries via supplied requirements.txt file
conda create --name fastapi --channel conda-forge --file requirements.txt

# Activate new environment
conda activate fastapi

# [if needed] Install git through conda
conda install git

# Copy and paste all files from ./starter to ./, then remove ./starter non-empty directory
cp -r ./starter/. ./
rm -r starter

# Initialize Git and DVC
git init
dvc init
```


## AWS Installation and Setup

First, run the following commands to install [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) on macOS:  

```shell
# Install AWS CLI on macOS
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg ./AWSCLIV2.pkg -target /
```

Second, create a S3 bucket via the following steps:  

* In the navigation bar in the Udacity classroom select **Open AWS Gateway** and then click **Open AWS Console**. You will not need the AWS Access Key ID or Secret Access Key provided here.
* From the Services drop down select S3 and then click Create bucket.
* Give your bucket a name, the rest of the options can remain at their default.

In order to use newly created S3 bucket from the AWS CLI installed from the first step, create an IAM user with the appropriate permissions via the following steps:  

* Sign in to the IAM console <a href="https://console.aws.amazon.com/iam/" target="_blank">here</a> or from the Services drop down on the upper navigation bar.
* In the left navigation bar select **Users**, then choose **Add user**.
* Give the user a name and select **Programmatic access**.
* In the permissions selector, search for S3 and give it **AmazonS3FullAccess**
* Tags are optional and can be skipped.
* After reviewing your choices, click create user. 
* To configure your AWS CLI, run ```aws configure``` in terminal, then type in the corresponding info from the `credential.csv` you just downloaded from the previous step.


## Github Actions Setup

* Setup GitHub Actions on your repository. Set up `python-version` inside of your `.github/workflows/[whatever-name-you-like].yml`
* Add your AWS credentials to the Action using the following format:
```shell
- name: Configure AWS credentials
   uses: aws-actions/configure-aws-credentials@v1
   with:
      aws-access-key-id: your_aws_access_key_id
      aws-secret-access-key: your_aws_secret_access_key
      aws-region: your_aws_region
```
* Set up <a href="https://github.com/iterative/setup-dvc" target="_blank">DVC in the action</a> and specify a command to `dvc pull` like the following:
```shell
- name: Setup DVC
   uses: iterative/setup-dvc@v1
- name: Pull data from DVC
   run: |
      dvc pull data -R
```
* Set up flake8 and pytest for lint check and unit testing.


## Data
Since the original raw data is messy, here I did some EDA using jupyter notebook, then the cleaned data is saved as csv file.
```shell
# Initialize DVC
dvc init

# Create a remote DVC named `census` and point it to S3 bucket
dvc remote add -d census s3://[your_s3_bucket_name]

# Add and push raw and clean data to remote S3 bucket
dvc add data/raw_data/raw_census.csv data/clean_data/clean_census.csv model/model.joblib model/lb.joblib model/one.joblib
dvc push
```


## Model

There are 3 main procedures: basic cleaning, model training and model inference. In order to run them separately, follow the next instructions:
```shell
# Execute basic cleaning
python main.py --action basic_cleaning

# Execute model training
python main.py --action model_training

# Execute model inference
python main.py --action model_inference
```

If you want to run the entire pipeline, use the following code:
```shell
# Execute entire ml pipeline
python main.py --action combo
# Or
python main.py
```


## API servc locally

In order to see results from your FastAPI locally, run this command:
```shell
uvicorn src.api:app --reload
```


## API Deployment - Heroku setup using Heroku CLI
First, create a free Heroku account. For the next steps, we will use the Heroku CLI to do setup.

* Create a new app named `udacity-fastapi`:
```shell
heroku create --app udacity-mlops-fastapi
```

* Install buildpack in order to use DVC in Heroku. Additionally, add heroku/python to your buildpacks run since this is Python application:
```shell
heroku buildpacks:clear # to clear previous configs
heroku buildpacks:add --index 1 heroku-community/apt # to enable heroku build packages in Aptfile
heroku buildpacks:add --index 2 heroku/python # to install python libraries in requirements.txt 
```

To specify a Python runtime for a new Heroku Python application, add a `runtime.txt` file to your app’s root directory that declares the exact version number to use:
```shell
touch runtime.txt; echo "python-3.8.12" >> runtime.txt
```

* In your root project folder create a file called `Aptfile` that specifies the release of DVC you want installed:
```shell
touch Aptfile; echo "https://github.com/iterative/dvc/releases/download/2.0.18/dvc_2.0.18_amd64.deb" >> Aptfile
```
Then, push all new changes to your remote github repo.
 
* Add the following code block to your `main.py`:
```shell
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
```

* Set up access to AWS on Heroku, if using the CLI:
```shell
heroku config:set AWS_ACCESS_KEY_ID=AKIAVDR6R626IQLV6HA4 AWS_SECRET_ACCESS_KEY=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
```

* Push code to Heroku:
```shell
git push heroku master
```


# Heroku API checkup

To check results returned from Heroku, run the following command:
```shell
python heroku_api.py
```