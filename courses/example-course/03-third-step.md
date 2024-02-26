_**LAST UPDATED:** 26/9/2023, by [Ran Yahalom](https://wix.slack.com/archives/D028P8YJY64)_

<!-- TOC -->
* [Creating your model class](#creating-your-model-class)
* [Methods you MUST implement](#methods-you-must-implement)
  * [`get_training_data(self)`](#gettrainingdataself)
  * [`schema(self)`](#schemaself)
  * [`fit(self, df, context, **kwargs)`](#fitself-df-context-kwargs)
  * [`predict(self, context, df)`](#predictself-context-df)
* [Additional methods you MAY also need to implement](#additional-methods-you-may-also-need-to-implement)
  * [`artifacts(self)`](#artifactsself)
  * [`load_context(self, context)`](#loadcontextself-context)
* [Putting all together](#putting-all-together)
* [Assignment](#assignment)
<!-- TOC -->

# Creating your model class
👉 In order to use ML platform to build, deploy, trigger and monitor your model, it must extend the [`wixml.model.BaseWixModel2`](https://github.com/wix-private/data-services/blob/master/ml-framework/wix-python-ml/wixml/model/base.py) class.

👉 This class defines the basic API required by ML platform to build, save, and deploy your model.

👉 In the ["Project Setup" lesson](https://github.com/wix-private/ds-ml-models/blob/master/ml-platform-course/01%20Project%20Setup/Project_setup.md) we saw an outline of what your model's class might look like:

```python
from wixml.model import BaseWixModel2
import pandas as pd
from mlflow.pyfunc import PythonModelContext
from wixml.model.schema import ModelSchema

class MyAmazingModel(BaseWixModel2):
    def get_training_data(self):
        ...

    def fit(self, df_train: pd.DataFrame, context: PythonModelContext = None, **kwargs):
        ...

    def predict(self, context, df_predict: pd.DataFrame) -> pd.DataFrame:
        ...

    def schema(self) -> ModelSchema:
        ...

    def artifacts(self) -> dict:
        ...

    def load_context(self, context: PythonModelContext):
        ...
```

⚠️ NOTE: we will often use the Elipsis (`...`) python constant as a placeholder to indicate a snippet of code that is omitted for clarity.

👉 In this lesson we'll gradually create an example model class named `FraudDetectionModel`, which uses a binary [Catboost classifier](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier) to infer if a Wix user (UUID) is a fraudster based on a collection of account / website / premium purchases related features:

```python
from wixml.model import BaseWixModel2
from catboost import CatBoostClassifier
...

# Define the catboost configuration parameters and probability threshold for making binary decisions (i.e. whether 
# the UUID is detected as a fraudster), which we obtained by conducting a hyperparameter tuning analysis:
CATBOOST_PARAMS = {
    'boosting_type': 'Ordered',
    'iterations': 91,
    'depth': 8,
    'l2_leaf_reg': 6.397093294988911,
    'learning_rate': 0.16335835748594907,
    'loss_function': 'Logloss',
    'max_ctr_complexity': 9,
    'verbose': True,
    'random_seed': 42
}
PROB_THRESHOLD_FOR_BIN_DECISIONS = 0.85
...

class FraudDetectionModel(BaseWixModel2):
    ...

    def __init__(self):
        self.clf = CatBoostClassifier(**CATBOOST_PARAMS)
        ...

    ...
```

# Methods you MUST implement
## `get_training_data(self)`
👉 This method should return the model's training set as a pandas dataframe.

👉 It is used during the build process of the model to save the training set for reproducibility and tracking.

👉 Assuming that the necessary training data for our `FraudDetectionModel` example is conveniently stored in a presto table, we could implement this method as follows:

```python
from wixml.model import BaseWixModel2
import pandas as pd
from wixds_general_utils.presto_utils import read_dataframe_from_db
...

# Define some more constants we will be using in our code:
TRAINING_DATA_PRESTO_TABLE = 'sandbox.ml_platform_course.model_creation_training_data'
...

class FraudDetectionModel(BaseWixModel2):
    ...

    def get_training_data(self) -> pd.DataFrame:
        """
        Fetches the training data from a presto table.
        @:returns The fetched data as a pandas DataFrame.
        """
        df_train = read_dataframe_from_db(f'select * from {TRAINING_DATA_PRESTO_TABLE}')
        return df_train

    ...
```

## `schema(self)`
👉 If you intend to invoke your model via ML platform, you must override this method to return a [`wixml.model.schema.ModelSchema`](https://github.com/wix-private/data-services/blob/master/ml-framework/wix-python-ml/wixml/model/schema.py) object that defines the I/O required for invoking your model.

👉 For our `FraudDetectionModel` example, let's define two dictionaries mapping the names of columns in our model's input and output data frames to their data types, and use them to create the `wixml.model.schema.ModelSchema` object that the `schema` will return:

```python
from wixml.model import BaseWixModel2
from wixml.model.schema import InputSchemaFeatureType, ModelSchema, InputSchema, OutputSchema, Feature, Prediction
...

# Define some more constants we will be using in our code:
...
LABEL_COL = 'label'
LEGIT_LABEL = 'Prob[LEGIT]'
FRAUD_LABEL = 'Prob[FRAUD]'
UUID_COL = 'uuid'
MSID_COL = 'msid'
MODEL_DECISION = 'is_predicted_as_fraud'

# Define a dict mapping the names of columns in our model's input data frame to their data type:
MODEL_INPUT_COLUMNS = {
    UUID_COL: InputSchemaFeatureType.STRING,
    MSID_COL: InputSchemaFeatureType.STRING,
    'declined_transactions_ratio': InputSchemaFeatureType.FLOAT,
    'is_high_decline_ratio': InputSchemaFeatureType.FLOAT,
    'num_card_holders_in_purchasing_history': InputSchemaFeatureType.FLOAT,
    'num_cards': InputSchemaFeatureType.FLOAT,
    'max_time_gap_in_min': InputSchemaFeatureType.FLOAT,
    'max_num_cards_per_invoice': InputSchemaFeatureType.FLOAT,
    'max_num_attempts_per_invoice': InputSchemaFeatureType.FLOAT,
    'is_active_partner': InputSchemaFeatureType.BOOLEAN,
    'browser_name': InputSchemaFeatureType.STRING,
    'main_industry': InputSchemaFeatureType.STRING,
    'msid_is_linked_to_blocked_account': InputSchemaFeatureType.STRING
}
# Define a dict mapping the names of columns in our model's output data frame to their data type:
MODEL_OUTPUT_COLUMNS = {
    LEGIT_LABEL: "STRING",
    FRAUD_LABEL: "STRING",
    MODEL_DECISION: "INT"
}


class FraudDetectionModel(BaseWixModel2):
    ...

    def __init__(self):
        ...

        self.features_used_to_fit_catboost = []  # This will be used to set a fixed order for the features in the data
        # frames that should be used for both fitting and prediction
        self.cat_bool_features = []  # Store the sublist of categorical and boolean features.

        for f_name, f_type in MODEL_INPUT_COLUMNS.items():
            if f_name not in [UUID_COL, MSID_COL]:
                self.features_used_to_fit_catboost.append(f_name)
                if f_type == InputSchemaFeatureType.STRING or f_type == InputSchemaFeatureType.BOOLEAN:
                    self.cat_bool_features.append(f_name)

    ...

    def schema(self) -> ModelSchema:
        """
        Use what we specified in the MODEL_INPUT_COLUMNS and MODEL_OUTPUT_COLUMNS dicts to return the model's 
        :py:mod:`wixml.model.schema.ModelSchema`.
        :returns The model's :py:mod:`wixml.model.schema.ModelSchema`.
        """
        return ModelSchema(
            input=InputSchema(features=[
                Feature(name=f_name, type=f_type, auto_extracted=False) for f_name, f_type in
                MODEL_INPUT_COLUMNS.items()
            ]),
            output=OutputSchema(predictions=[
                Prediction(name=p_name, type=p_type) for p_name, p_type in MODEL_OUTPUT_COLUMNS.items()
            ])
        )

    ...

```

👉 Notice that we now also use the `MODEL_INPUT_COLUMNS` dictionary (from within the `__init__()` method) to:
- Set a fixed order for the features columns, so we can be sure it is the same for both training and prediction.
- Set the list of features that the Catboost classifier should regard as categorical.

## `fit(self, df, context, **kwargs)`
👉 This method should trigger the model's fit process.

👉 It is called during the build process by the ML platform which passes it the data frame returned by the `get_training_data(self)` method.

👉 If you set your model's `fit_context` class member to `True`, ML platform will also instantiate the `context` argument passed to this function (more on that ahead).

👉 After the call to this method completes, it must be possible to pickle your model's instance using the [`cloudpickle` pickling library](https://pypi.org/project/cloudpickle/). You can verify this by invoking the `wixds_general_utils.utils.pickle_obj()` utility function on your trained model's instance.

👉 Going back to our `FraudDetectionModel` example, let's assume that our training data frame includes a column representing the class label of each row and use it (along with our model's `features_used_to_fit_catboost`, `cat_bool_features` and `clf` instance attributes) to fit the Catboost classifier:
```python
from wixml.model import BaseWixModel2
import pandas as pd
from mlflow.pyfunc import PythonModelContext
...

# Define some more constants we will be using in our code:
LABEL_COL = 'label'
...

class FraudDetectionModel(BaseWixModel2):
    ...

    def fit(self, df_train: pd.DataFrame, context: PythonModelContext = None, **kwargs):
        """
        This method instantiates and fits the model's Catboost classifier.
        
        :param pd.DataFrame df_train: the training data returned by the self.get_training_data() method.
        :param PythonModelContext context: object containing additional context. Since we defined `fit_context` = True,
        this object will include a property attribute named "artifacts" which is a dictionary similar to the one 
        returned by the model's artifacts() method except that paths point to the local copies of the loaded artifacts.
        :param kwargs: additional keyword arguments that your fit function can accept. This is here mainly for 
        compatability, because the ML platform will not pass these arguments to your fit function when fitting your
        model.
        """
        y_train = df_train[LABEL_COL]
        # Fit the Catboost classifier:
        X_train = df_train[self.features_used_to_fit_catboost]
        cat_and_bool_col_inds = [X_train.columns.get_loc(col) for col in self.cat_bool_features]
        self.clf.fit(X_train, y_train, cat_features=cat_and_bool_col_inds)

```

## `predict(self, context, df)`
👉 This is your model's inference method which will be exposed after deployment and invoked whenever your model is triggered.

👉 When invoked, ML platform will pass it both a `context` object and a pandas data frame containing the input to your model.

👉 This method must return a pandas dataframe with column names and dtypes corresponding to the `OutputSchema` returned by the `schema()` method.

👉 Now let's implement a `predict` method for our `FraudDetectionModel` which uses the trained catboost classifier to calculate and return the per-class probabilities, along with the infered class, for each row in the prediction data frame:

```python
from wixml.model import BaseWixModel2
import pandas as pd
from mlflow.pyfunc import PythonModelContext
...

PROB_THRESHOLD_FOR_BIN_DECISIONS = 0.85

# Define some more constants we will be using in our code:
LEGIT_LABEL = 'Prob[LEGIT]'
FRAUD_LABEL = 'Prob[FRAUD]'
MODEL_DECISION = 'is_predicted_as_fraud'
...

class FraudDetectionModel(BaseWixModel2):
    ...

    def predict(self, context: PythonModelContext, df_predict: pd.DataFrame) -> pd.DataFrame:
        """
        This method uses the stored self.features_used_to_fit_catboost list to obtain the appropriate data frame
        for prediction and then invoke the model's trained catboost classifier.
        :param pd.DataFrame df_predict: the data frame outputted by the before_predict_callback method.
        :param PythonModelContext context: object containing additional context.
        :return: The data frame with the prediction results.
        """
        X_pred = df_predict[self.features_used_to_fit_catboost]
        proba = self.clf.predict_proba(X_pred)
        prediction_result_df = pd.DataFrame(proba, columns=[LEGIT_LABEL, FRAUD_LABEL])
        prediction_result_df[MODEL_DECISION] = (prediction_result_df[FRAUD_LABEL] >=
                                                PROB_THRESHOLD_FOR_BIN_DECISIONS).astype(int)
        return prediction_result_df
```

# Additional methods you MAY also need to implement
## `artifacts(self)`
👉 This method returns a dictionary of paths to external resources (aka "artifacts") that your model needs to use, e.g. large pickled objects or files.

👉 These paths should be URIs of either local or remote AWS S3 files that are accessible to the ML Platform. For example, if the dictionary returned by your implementation contains an item whose value is `"s3://wix-ds-models/groupen/groupen-cv/v1.6/full_80.pth"`, the ML Platform will attempt to fetch the `full_80.pth` file located under the AWS S3 bucket `"s3://wix-ds-models/groupen/groupen-cv/v1.6`.

👉 When your model is triggered, ML platform will load the artifacts and then add a dictionary of paths, each pointing to a local copy of one of the external resources, to the `context` argument of the `predict` method.

👉 If you also want the artifacts to be passed to the `context` argument of your `fit()` method, you **MUST** initialize the model class attribute `fit_context` to `True`.

👉 To demonstrate this, let's assume that we want our `FraudDetectionModel` to make use of a known fraud pattern where fraudsters sometimes purchase premium plans that are rarely purchased by new Wix users or by users that still haven't created any website. This is often the case when fraudsters are doing the purchase only to check if a stolen credit card can be used (aka "credit card testing"). To do this we want our `FraudDetectionModel` to extract two additional features that encode the fraud pattern by scoring the rareness of the {plan, cycle, business type} triplet corresponding to the premium purchase transactions of each UUID, given:
1. The age of the UUID's account, i.e. how many hours passed between opening the account and the time of the transaction.
2. The fact that the UUID doesn't have any websites.
   In order to calculate these two rareness score features, our model needs access to the distribution of account ages and "websiteless" UUIDs for each {plan, cycle, business type} triplet. Without going into the design considerations, let's assume we want to store this information as a pandas dataframe containing a row per unique combination of the account age, triplet, and "has site" indicator, each appearing in a separate column. The data frame also contains a column with the list of UUIDs that are characterized by the row's unique combination. Since we continuously update the distribution stored in this data frame as more users join Wix, it will continuously grow in size so we cannot include it in our models production directory. A good solution is to maintain this data frame as a compressed file, e.g. using the "feather" binary format, on S3 and load it as an external artifact. Adding this to the `FraudDetectionModel` class involves the following modifications:

```python
from wixml.model import BaseWixModel2
import pandas as pd
from mlflow.pyfunc import PythonModelContext
from wixml.model.schema import InputSchemaFeatureType
from pandas.core.groupby.generic import DataFrameGroupBy
from wixds_general_utils.utils import replace_NA_strings_with_None
import json
...

UUID_COL = 'uuid'
MSID_COL = 'msid'
RARE_TRIPLET_GIVEN_NO_WEBSITE_SCORE_COL = 'rareness_of_product_cycle_business_type_triplet_when_no_site_score'
RARE_TRIPLET_GIVEN_AGE_SCORE_COL = 'rareness_of_product_cycle_business_type_triplet_given_age_score'
HAS_SITE_COL = 'has_site'
ACCOUNT_AGE_COL = 'account_age_at_first_purch_in_minutes'
TRIPLETS_COL = 'product_cycle_business_type_triplets'
COMBINATIONS_FILE_ARTIFACT_NAME = "feather_formatted_combinations_file"
COMBINATIONS_FILE_S3_URI = "s3://wix-ds-models/ml-platform-course/age_triplets_site_combinations.feather"

MODEL_INPUT_COLUMNS = {
    UUID_COL: InputSchemaFeatureType.STRING,
    MSID_COL: InputSchemaFeatureType.STRING,
    'declined_transactions_ratio': InputSchemaFeatureType.FLOAT,
    'is_high_decline_ratio': InputSchemaFeatureType.FLOAT,
    'num_card_holders_in_purchasing_history': InputSchemaFeatureType.FLOAT,
    'num_cards': InputSchemaFeatureType.FLOAT,
    'max_time_gap_in_min': InputSchemaFeatureType.FLOAT,
    'max_num_cards_per_invoice': InputSchemaFeatureType.FLOAT,
    'max_num_attempts_per_invoice': InputSchemaFeatureType.FLOAT,
    'is_active_partner': InputSchemaFeatureType.BOOLEAN,
    'browser_name': InputSchemaFeatureType.STRING,
    'main_industry': InputSchemaFeatureType.STRING,
    'msid_is_linked_to_blocked_account': InputSchemaFeatureType.STRING,
    HAS_SITE_COL: InputSchemaFeatureType.BOOLEAN,
    ACCOUNT_AGE_COL: InputSchemaFeatureType.FLOAT,
    TRIPLETS_COL: InputSchemaFeatureType.STRING
}
...

class FraudDetectionModel(BaseWixModel2):
    fit_context = True  # We need to do this so that the model context will be passed to the fit() method when ML 
    # platform builds the model. This context object will include a property attribute named 
    # "artifacts" which is a dictionary similar to the one returned by the model's artifacts()
    # method except that paths point to the local copies of the loaded artifacts.

    def __init__(self):
        ...

        self.features_used_to_fit_catboost += [RARE_TRIPLET_GIVEN_AGE_SCORE_COL,
                                               RARE_TRIPLET_GIVEN_NO_WEBSITE_SCORE_COL]
    ...

    def artifacts(self):
        return {
            COMBINATIONS_FILE_ARTIFACT_NAME: COMBINATIONS_FILE_S3_URI
        }

    def fit(self, df_train: pd.DataFrame, context: PythonModelContext = None, **kwargs):
        ...
        triplet_groups = load_triplet_groups_from_feather(context.artifacts.get(COMBINATIONS_FILE_ARTIFACT_NAME))
        df_train = add_rareness_scores(df_train, triplet_groups)
        ...

    def predict(self, context: PythonModelContext, df_predict: pd.DataFrame) -> pd.DataFrame:
        ...
        triplet_groups = load_triplet_groups_from_feather(context.artifacts.get(COMBINATIONS_FILE_ARTIFACT_NAME))
        df_predict = add_rareness_scores(df_predict, triplet_groups)
        ...


def load_triplet_groups_from_feather(filepath_to_feather_formatted_df) -> DataFrameGroupBy:
    triplet_counts_df = pd.read_feather(filepath_to_feather_formatted_df)
    # Unmarshal the set of uuids per triplet from the JSON string we used to store it in the feather encoded
    # data frame:
    triplet_counts_df[UUID_COL] = triplet_counts_df[UUID_COL].transform(lambda x: json.loads(x))
    triplet_groups = triplet_counts_df.groupby(TRIPLETS_COL)
    return triplet_groups

def add_rareness_scores(df, triplet_groups):
    uuid_age_triplets_site_data_df = replace_NA_strings_with_None(
        df[UUID_COL, ACCOUNT_AGE_COL, TRIPLETS_COL, HAS_SITE_COL])
    rareness_scores_df = uuid_age_triplets_site_data_df.apply(calc_rareness_per_uuid,
                                                              triplet_groups=triplet_groups,
                                                              axis=1)
    return df.join(rareness_scores_df.set_index(UUID_COL, drop=True), on=UUID_COL)


def calc_rareness_per_uuid(uuid_row: pd.Series, triplet_groups: DataFrameGroupBy):
    age = uuid_row[ACCOUNT_AGE_COL]
    has_site = uuid_row[HAS_SITE_COL]
    age_percentile = 0
    frac_no_site = 0
    triplets_set = uuid_row[TRIPLETS_COL]
    for triplet in triplets_set:
        if triplet in triplet_groups.groups:  # The effect of this condition is that a triplet not seen in the
            # training data will be regarded as the rarest triplet possible.
            triplet_group = triplet_groups.get_group(triplet)
            if pd.notna(age):
                known_ages_for_this_triplet = triplet_group[ACCOUNT_AGE_COL]
                age_percentile += (known_ages_for_this_triplet < age).mean()
            if not has_site:
                has_site_for_this_triplet_counts = triplet_group[HAS_SITE_COL].value_counts(normalize=True)
                frac_no_site += has_site_for_this_triplet_counts[0] if 0 in has_site_for_this_triplet_counts else 0

    n = len(triplets_set)
    rareness_features = {
        RARE_TRIPLET_GIVEN_AGE_SCORE_COL: 1 if n == 0 else 1 - (age_percentile / n),
        RARE_TRIPLET_GIVEN_NO_WEBSITE_SCORE_COL: 0 if (has_site or n == 0) else 1 - (frac_no_site / n),
        UUID_COL: uuid_row[UUID_COL]
    }
    return pd.Series(rareness_features)

```

## `load_context(self, context)`
👉 This method is called by the ML platform during deployment to initialize your model instance before calling its `predict()` method.

👉 Similarly to the `predict()` method, the `context` argument will contain the artifacts attribute.

👉 For example, it would be more efficient for our `FraudDetectionModel` to load the triplet groups from a large feather file only once during deployment from this method rather than do it each time the `predict()` method is invoked. We can do this as follows:

```python
from wixml.model import BaseWixModel2
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from mlflow.pyfunc import PythonModelContext
...

COMBINATIONS_FILE_ARTIFACT_NAME = "feather_formatted_combinations_file"
COMBINATIONS_FILE_S3_URI = "s3://wix-ds-models/ml-platform-course/age_triplets_site_combinations.feather"
...

class FraudDetectionModel(BaseWixModel2):
    ...

    def load_context(self, context: PythonModelContext):
       """
       This method is called by the ML platform during deployment to initialize your model instance before calling its `predict()` method. We load the triplet groups from the large feather file here so it happens only once during deployment instead of repeatedly each time the `predict()` method is invoked. 
       """ 
       self.triplet_groups = load_triplet_groups_from_feather(context.artifacts.get(COMBINATIONS_FILE_ARTIFACT_NAME))

    def predict(self, context: PythonModelContext, df_predict: pd.DataFrame) -> pd.DataFrame:
        ...
        df_predict = add_rareness_scores(df_predict, self.triplet_groups)
        ...

def load_triplet_groups_from_feather(filepath_to_feather_formatted_df) -> DataFrameGroupBy:
    ...

def add_rareness_scores(df, triplet_groups):
    ...

```

# Putting all together
👉 If you followed all the previous steps in this lesson correctly, you should now have a fully functional version of our example `FraudDetectionModel` that looks like this:

```python
from wixml.model import BaseWixModel2
from catboost import CatBoostClassifier
import pandas as pd
from wixds_general_utils.presto_utils import read_dataframe_from_db
from mlflow.pyfunc import PythonModelContext
from wixml.model.schema import InputSchemaFeatureType, ModelSchema, InputSchema, OutputSchema, Feature, Prediction
from pandas.core.groupby.generic import DataFrameGroupBy
from wixds_general_utils.utils import replace_NA_strings_with_None
import json

# Define the catboost configuration parameters and probability threshold for making binary decisions (i.e. whether 
# the model inferred that a UUID is a fraudster), which we obtained by conducting a hyperparameter tuning analysis:
CATBOOST_PARAMS = {
    'boosting_type': 'Ordered',
    'iterations': 91,
    'depth': 8,
    'l2_leaf_reg': 6.397093294988911,
    'learning_rate': 0.16335835748594907,
    'loss_function': 'Logloss',
    'max_ctr_complexity': 9,
    'verbose': True,
    'random_seed': 42
}
PROB_THRESHOLD_FOR_BIN_DECISIONS = 0.85

# Define some more constants we will be using in our code:
TRAINING_DATA_PRESTO_TABLE = 'sandbox.ml_platform_course.model_creation_training_data'
LABEL_COL = 'label'
LEGIT_LABEL = 'Prob[LEGIT]'
FRAUD_LABEL = 'Prob[FRAUD]'
UUID_COL = 'uuid'
MSID_COL = 'msid'
MODEL_DECISION = 'is_predicted_as_fraud'
RARE_TRIPLET_GIVEN_NO_WEBSITE_SCORE_COL = 'rareness_of_product_cycle_business_type_triplet_when_no_site_score'
RARE_TRIPLET_GIVEN_AGE_SCORE_COL = 'rareness_of_product_cycle_business_type_triplet_given_age_score'
HAS_SITE_COL = 'has_site'
ACCOUNT_AGE_COL = 'account_age_at_first_purch_in_minutes'
TRIPLETS_COL = 'product_cycle_business_type_triplets'
COMBINATIONS_FILE_ARTIFACT_NAME = "feather_formatted_combinations_file"
COMBINATIONS_FILE_S3_URI = "s3://wix-ds-models/ml-platform-course/age_triplets_site_combinations.feather"

# Define a dict mapping the names of columns in our model's input data frame to their data type:
MODEL_INPUT_COLUMNS = {
    UUID_COL: InputSchemaFeatureType.STRING,
    MSID_COL: InputSchemaFeatureType.STRING,
    'declined_transactions_ratio': InputSchemaFeatureType.FLOAT,
    'is_high_decline_ratio': InputSchemaFeatureType.FLOAT,
    'num_card_holders_in_purchasing_history': InputSchemaFeatureType.FLOAT,
    'num_cards': InputSchemaFeatureType.FLOAT,
    'max_time_gap_in_min': InputSchemaFeatureType.FLOAT,
    'max_num_cards_per_invoice': InputSchemaFeatureType.FLOAT,
    'max_num_attempts_per_invoice': InputSchemaFeatureType.FLOAT,
    'is_active_partner': InputSchemaFeatureType.BOOLEAN,
    'browser_name': InputSchemaFeatureType.STRING,
    'main_industry': InputSchemaFeatureType.STRING,
    'msid_is_linked_to_blocked_account': InputSchemaFeatureType.STRING,
    HAS_SITE_COL: InputSchemaFeatureType.BOOLEAN,
    ACCOUNT_AGE_COL: InputSchemaFeatureType.FLOAT,
    TRIPLETS_COL: InputSchemaFeatureType.STRING
}
# Define a dict mapping the names of columns in our model's output data frame to their data type:
MODEL_OUTPUT_COLUMNS = {
    LEGIT_LABEL: "STRING",
    FRAUD_LABEL: "STRING",
    MODEL_DECISION: "INT"
}

class FraudDetectionModel(BaseWixModel2):
    fit_context = True  # We need to do this so that the model context will be passed to the fit() method when ML 
    # platform builds the model. This context object will include a property attribute named 
    # "artifacts" which is a dictionary similar to the one returned by the model's artifacts()
    # method except that paths point to the local copies of the loaded artifacts.

    def __init__(self):
        self.clf = CatBoostClassifier(**CATBOOST_PARAMS)
        self.features_used_to_fit_catboost = []  # This will be used to set a fixed order for the features in the data
        # frames that should be used for both fitting and prediction
        self.cat_bool_features = []  # Store the sublist of categorical and boolean features.

        for f_name, f_type in MODEL_INPUT_COLUMNS.items():
            if f_name not in [UUID_COL, MSID_COL]:
                self.features_used_to_fit_catboost.append(f_name)
                if f_type == InputSchemaFeatureType.STRING or f_type == InputSchemaFeatureType.BOOLEAN:
                    self.cat_bool_features.append(f_name)

        self.features_used_to_fit_catboost += [RARE_TRIPLET_GIVEN_AGE_SCORE_COL,
                                               RARE_TRIPLET_GIVEN_NO_WEBSITE_SCORE_COL]

    def get_training_data(self) -> pd.DataFrame:
        """
        Fetches the training data from a presto table.
        @:returns The fetched data as a pandas DataFrame.
        """
        df_train = read_dataframe_from_db(f'select * from {TRAINING_DATA_PRESTO_TABLE}')
        return df_train

    def schema(self) -> ModelSchema:
        """
        Use what we specified in the MODEL_INPUT_COLUMNS and MODEL_OUTPUT_COLUMNS dicts to return the model's 
        :py:mod:`wixml.model.schema.ModelSchema`.
        :returns The model's :py:mod:`wixml.model.schema.ModelSchema`.
        """
        return ModelSchema(
            input=InputSchema(features=[
                Feature(name=f_name, type=f_type, auto_extracted=False) for f_name, f_type in
                MODEL_INPUT_COLUMNS.items()
            ]),
            output=OutputSchema(predictions=[
                Prediction(name=p_name, type=p_type) for p_name, p_type in MODEL_OUTPUT_COLUMNS.items()
            ])
        )

    def artifacts(self):
        return {
            COMBINATIONS_FILE_ARTIFACT_NAME: COMBINATIONS_FILE_S3_URI
        }

    def load_context(self, context: PythonModelContext):
       """
       This method is called by the ML platform during deployment to initialize your model instance before calling its `predict()` method. We load the triplet groups from the large feather file here, so it happens only once during deployment instead of repeatedly each time the `predict()` method is invoked. 
       """
       self.triplet_groups = load_triplet_groups_from_feather(context.artifacts.get(COMBINATIONS_FILE_ARTIFACT_NAME))

    def fit(self, df_train: pd.DataFrame, context: PythonModelContext = None, **kwargs):
        """
        This method instantiates and fits the model's Catboost classifier.
        
        :param pd.DataFrame df_train: the training data returned by the self.get_training_data() method.
        :param PythonModelContext context: object containing additional context. Since we defined `fit_context` = True,
        this object will include a property attribute named "artifacts" which is a dictionary similar to the one 
        returned by the model's artifacts() method except that paths point to the local copies of the loaded artifacts.
        :param kwargs: additional keyword arguments that your fit function can accept. This is here mainly for 
        compatability, because the ML platform will not pass these arguments to your fit function when fitting your
        model.
        """
        triplet_groups = load_triplet_groups_from_feather(context.artifacts.get(COMBINATIONS_FILE_ARTIFACT_NAME))
        df_train = add_rareness_scores(df_train, triplet_groups)
        y_train = df_train[LABEL_COL]
        # Fit the Catboost classifier:
        X_train = df_train[self.features_used_to_fit_catboost]
        cat_and_bool_col_inds = [X_train.columns.get_loc(col) for col in self.cat_bool_features]
        self.clf.fit(X_train, y_train, cat_features=cat_and_bool_col_inds)

    def predict(self, context: PythonModelContext, df_predict: pd.DataFrame) -> pd.DataFrame:
        """
        This method uses the stored self.features_used_to_fit_catboost list to obtain the appropriate data frame
        for prediction and then invoke the model's trained catboost classifier.
        :param pd.DataFrame df_predict: the data frame outputted by the before_predict_callback method.
        :param PythonModelContext context: object containing additional context.
        :return: The data frame with the prediction results.
        """
        df_predict = add_rareness_scores(df_predict, self.triplet_groups)
        X_pred = df_predict[self.features_used_to_fit_catboost]
        proba = self.clf.predict_proba(X_pred)
        prediction_result_df = pd.DataFrame(proba, columns=[LEGIT_LABEL, FRAUD_LABEL])
        prediction_result_df[MODEL_DECISION] = (prediction_result_df[FRAUD_LABEL] >=
                                                PROB_THRESHOLD_FOR_BIN_DECISIONS).astype(int)
        return prediction_result_df

def load_triplet_groups_from_feather(filepath_to_feather_formatted_df) -> DataFrameGroupBy:
    triplet_counts_df = pd.read_feather(filepath_to_feather_formatted_df)
    # Unmarshal the set of uuids per triplet from the JSON string we used to store it in the feather encoded
    # data frame:
    triplet_counts_df[UUID_COL] = triplet_counts_df[UUID_COL].transform(lambda x: json.loads(x))
    triplet_groups = triplet_counts_df.groupby(TRIPLETS_COL)
    return triplet_groups

def add_rareness_scores(df, triplet_groups):
    uuid_age_triplets_site_data_df = replace_NA_strings_with_None(
        df[UUID_COL, ACCOUNT_AGE_COL, TRIPLETS_COL, HAS_SITE_COL])
    rareness_scores_df = uuid_age_triplets_site_data_df.apply(calc_rareness_per_uuid,
                                                              triplet_groups=triplet_groups,
                                                              axis=1)
    return df.join(rareness_scores_df.set_index(UUID_COL, drop=True), on=UUID_COL)

def calc_rareness_per_uuid(uuid_row: pd.Series, triplet_groups: DataFrameGroupBy):
    age = uuid_row[ACCOUNT_AGE_COL]
    has_site = uuid_row[HAS_SITE_COL]
    age_percentile = 0
    frac_no_site = 0
    triplets_set = uuid_row[TRIPLETS_COL]
    for triplet in triplets_set:
        if triplet in triplet_groups.groups:  # The effect of this condition is that a triplet not seen in the
            # training data will be regarded as the rarest triplet possible.
            triplet_group = triplet_groups.get_group(triplet)
            if pd.notna(age):
                known_ages_for_this_triplet = triplet_group[ACCOUNT_AGE_COL]
                age_percentile += (known_ages_for_this_triplet < age).mean()
            if not has_site:
                has_site_for_this_triplet_counts = triplet_group[HAS_SITE_COL].value_counts(normalize=True)
                frac_no_site += has_site_for_this_triplet_counts[0] if 0 in has_site_for_this_triplet_counts else 0
                
    n = len(triplets_set)
    rareness_features = {
        RARE_TRIPLET_GIVEN_AGE_SCORE_COL: 1 if n == 0 else 1 - (age_percentile / n),
        RARE_TRIPLET_GIVEN_NO_WEBSITE_SCORE_COL: 0 if (has_site or n == 0) else 1 - (frac_no_site / n),
        UUID_COL: uuid_row[UUID_COL]
    }
    return pd.Series(rareness_features)

```

# Assignment
👉 Set up a new project for the above `FraudDetectionModel` example as follows:
   - Review the ["Project Setup" lesson](https://github.com/wix-private/ds-ml-models/blob/master/ml-platform-course/01%20Project%20Setup/Project_setup.md) to make sure that you follow all the necessary guidelines.
   - Name the project directory by "<MY_WIX_USERNAME>-ml-platform-course", after replacing "<MY_WIX_USERNAME>" with your Wix username.

👉 Submit a copy of the project directory you just set up to complete this assignment.

👏 Congratulations!!! You are now ready to continue to the next lesson and learn how to build your model on the ML platform. 

