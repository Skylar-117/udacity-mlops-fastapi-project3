Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.


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
* To configure your AWS CLI, run ```aws configure``` in terminal, then type in the corresponding info from the credential.csv you just downloaded from the previous step.


## Github Actions Setup

Create a YAML file named `test.yaml` under `./github/workflows` directory:  

```shell
# [if necessary] Create a `.github/workflows` directory inside your local Git repository if it does not already exist
mkdir .github/workflows && cd .github/workflows

# In the `.github/workflows` directory, create a file named test.yaml
touch test.yaml
```  

* Setup GitHub Actions on your repository. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
   * Make sure you set up the GitHub Action to have the same version of Python as you used in development.
* Add your <a href="https://github.com/marketplace/actions/configure-aws-credentials-action-for-github-actions" target="_blank">AWS credentials to the Action</a>.
* Set up <a href="https://github.com/iterative/setup-dvc" target="_blank">DVC in the action</a> and specify a command to `dvc pull`.


## Data
Since the original raw data is messy, here I did some EDA using jupyter notebook, then the cleaned data is saved as csv file.
```shell
# Initialize DVC
dvc init

# Create a remote DVC named `census` and point it to S3 bucket
dvc remote add -d census s3://udacity-mlops-fastapi/data

# Add and push raw and clean data to remote S3 bucket
dvc add data/raw_data/raw_census.csv data/clean_data/clean_census.csv
dvc push
```


## Model

* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
   * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.


## API Creation

* Create a RESTful API using FastAPI this must implement:
   * GET on the root giving a welcome message.
   * POST that does model inference.
   * Type hinting must be used.
   * Use a Pydantic model to ingest the body from POST. This model should contain an example.
    * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).


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
heroku config:set AWS_ACCESS_KEY_ID=ASIAVDR6R626OPFTBTYS AWS_SECRET_ACCESS_KEY=OWoS0uGvMC36vUehmDHx2sFj2NSt71ZmOu0g47wZ
```

* Push code to Heroku:
```shell
git push heroku master
```


* Create a new app and have it deployed from your GitHub repository.
   * Enable automatic deployments that only deploy if your continuous integration passes.
   * Hint: think about how paths will differ in your local environment vs. on Heroku.
   * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Set up DVC on Heroku using the instructions contained in the starter directory.
* Write a script that uses the requests module to do one POST on your live API.
