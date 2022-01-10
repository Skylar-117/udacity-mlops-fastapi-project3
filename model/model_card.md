# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a sklearn Random Forest Classifier with custom model parameters defined inside utils module.

## Intended Use
The model is used to predict the salary of a person based on a set of attributes about his/her financials.

## Training Data
Data source is from https://archive.ics.uci.edu/ml/datasets/census+income. 80% of the original data is used for training purpose. For categorical features, one hot encoding method is used to convert each category to 0/1 values, which ends up making the original dataset sparse. For numerical features, no advanced encoding is used.

## Evaluation Data
Data source is from https://archive.ics.uci.edu/ml/datasets/census+income. 20% of the original data is used for validation purpose.

## Metrics
The model was evaluated based on accuracy which is around 0.875.  
The model was evaluated based on recall which is around 0.615.  
The model was evaluated based on precision which is around 0.833.  
As we can see, recall is lower than accuracy and precision.  

## Ethical Considerations
Dataset contains demographic data like race, gender and origin country. This might drive to a model that discriminate people a bit. The model should be used with caution.

## Caveats and Recommendations
Further work needed to evaluate across a spectrum of genders.
