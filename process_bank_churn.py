import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler,PolynomialFeatures, OrdinalEncoder,LabelEncoder
from typing import Dict

def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Removes unnecessary columns such as 'id' and 'CustomerId'."""
    
    return df.drop(columns=['id', 'CustomerId'], errors='ignore')

def split_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the dataset into training and validation sets, stratified by the target variable."""
    
    return train_test_split(df, test_size=0.18, random_state=28, stratify=df[target_col])

def encode_categorical_features(train_df: pd.DataFrame, val_df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], OneHotEncoder]:
    """Encodes categorical features using OneHotEncoder with training fit and transformation on all sets."""
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[categorical_cols])

    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    train_encoded = pd.DataFrame(encoder.transform(train_df[categorical_cols]), columns=encoded_cols, index=train_df.index)
    val_encoded = pd.DataFrame(encoder.transform(val_df[categorical_cols]), columns=encoded_cols, index=val_df.index)

    train_df = train_df.drop(columns=categorical_cols).join(train_encoded)
    val_df = val_df.drop(columns=categorical_cols).join(val_encoded)

    return train_df, val_df, encoded_cols, encoder

def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates additional features based on numerical attributes and encodes a binned version of 'Age'."""
    
    df = df.copy()
    
    df["Balance/EstimatedSalary"] = df["Balance"] / df["EstimatedSalary"]
    df['Age/Salary'] = df['Age'] / df['EstimatedSalary']
    df['Balance/Product'] = df['Balance'] / df['NumOfProducts']
    df["Age-Tenure"] = df["Age"] - df["Tenure"]
    df["isActiveMember+HasCrCard"] = df["IsActiveMember"] + df["HasCrCard"]
    df['MoreThanOneProduct'] = (df['NumOfProducts'] > 1).astype(int)
    df['Balance_flag'] = (df['Balance'] > 0).astype(int)
    df["Tenure/Age"] = df["Tenure"] / df["Age"]
    
    return df

def generate_polynomial_features(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: List[str], degree: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], PolynomialFeatures]:
    """Creates polynomial features for numeric columns using PolynomialFeatures."""
    
    poly = PolynomialFeatures(degree=degree)
    poly.fit(train_df[numeric_cols])
    
    poly_cols = list(poly.get_feature_names_out(numeric_cols))

    train_poly_df = pd.DataFrame(poly.transform(train_df[numeric_cols]), columns=poly_cols, index=train_df.index)
    val_poly_df = pd.DataFrame(poly.transform(val_df[numeric_cols]), columns=poly_cols, index=val_df.index)

    train_df = train_df.drop(columns=numeric_cols).join(train_poly_df)
    val_df = val_df.drop(columns=numeric_cols).join(val_poly_df)

    return train_df, val_df, poly_cols, poly

def scale_numeric_features(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scales numeric features using StandardScaler with training fit and transformation on all sets."""
    
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])

    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])

    return train_df, val_df, scaler

def preprocess_data(df: pd.DataFrame, scaler_numeric: bool = True) -> Dict[str, Union[pd.DataFrame, pd.Series, OneHotEncoder, PolynomialFeatures, StandardScaler]]:
    """Runs the full data preprocessing pipeline for train and validation datasets."""
    
    df = remove_unnecessary_columns(df)

    train_df, val_df = split_data(df, target_col='Exited')

    train_targets = train_df.pop('Exited')
    val_targets = val_df.pop('Exited')

    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()

    train_df = create_new_features(train_df)
    val_df = create_new_features(val_df)
    
    train_df.drop(columns=['Surname'], inplace=True, errors='ignore')
    val_df.drop(columns=['Surname'], inplace=True, errors='ignore')

    categorical_cols = train_df.select_dtypes(include='object').columns.tolist()
    categorical_cols.append('isActiveMember+HasCrCard')
    train_df, val_df, encoded_cols, encoder = encode_categorical_features(train_df, val_df, categorical_cols)
   
    new_numeric_features = ["Balance/EstimatedSalary", "Age/Salary" ,"Age-Tenure", "Balance/Product", "Tenure/Age"]
    numeric_cols.extend(new_numeric_features)  
    
    train_df, val_df, poly_cols, poly = generate_polynomial_features(train_df, val_df, numeric_cols)

    if '1' in poly_cols:
        poly_cols.remove('1')

    if scaler_numeric:
        train_df, val_df, scaler = scale_numeric_features(train_df, val_df, poly_cols)

    feature_cols = poly_cols + encoded_cols + ['Balance_flag', 'MoreThanOneProduct']

    train_df = train_df[feature_cols]
    val_df = val_df[feature_cols]

    return {
        'train_X': train_df,
        'train_y': train_targets,
        'val_X': val_df,
        'val_y': val_targets,
        'encoder': encoder,
        'poly': poly,
        'scaler': scaler if scaler_numeric else None,
        'input_cols': train_df.columns
    }


def preprocess_new_data(test_df: pd.DataFrame, encoder: OneHotEncoder, poly: PolynomialFeatures, scaler: StandardScaler, scaler_numeric: bool = True) -> Dict[str, pd.DataFrame]:
    """Runs the full data preprocessing pipeline for test dataset."""
    
    test_df = remove_unnecessary_columns(test_df)

    numeric_cols = test_df.select_dtypes(include=np.number).columns.tolist()

    test_df = create_new_features(test_df)

    test_df.drop(columns=['Surname'], inplace=True, errors='ignore')
    
    categorical_cols = test_df.select_dtypes(include='object').columns.tolist()
    categorical_cols.append('isActiveMember+HasCrCard')
    
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    test_encoded = pd.DataFrame(encoder.transform(test_df[categorical_cols]), columns=encoded_cols, index=test_df.index)
    test_df = test_df.drop(columns=categorical_cols).join(test_encoded)
   
    new_numeric_features = ["Balance/EstimatedSalary", "Age/Salary" ,"Age-Tenure", "Balance/Product", "Tenure/Age"]
    numeric_cols.extend(new_numeric_features)  
    
    poly_cols = list(poly.get_feature_names_out(numeric_cols))
    test_poly_df = pd.DataFrame(poly.transform(test_df[numeric_cols]), columns=poly_cols, index=test_df.index)
    test_df = test_df.drop(columns=numeric_cols).join(test_poly_df)
    if '1' in poly_cols:
        poly_cols.remove('1')

    if scaler_numeric:
        test_df[poly_cols] = scaler.transform(test_df[poly_cols])

    feature_cols = poly_cols + encoded_cols + ['Balance_flag', 'MoreThanOneProduct']

    test_df = test_df[feature_cols]

    return {
        'test_X': test_df
    }