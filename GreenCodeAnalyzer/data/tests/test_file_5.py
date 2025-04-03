import pandas as pd
import numpy as np

def female_bias_via_undersampling(X_train, y_train):
    train_data = X_train.copy()
    train_data['checked'] = y_train

    # Separate training data by gender and checked status.
    male_checked = train_data[(train_data['persoon_geslacht_vrouw'] == 0) & (train_data['checked'] == 1)]
    male_not_checked = train_data[(train_data['persoon_geslacht_vrouw'] == 0) & (train_data['checked'] == 0)]

    female_checked = train_data[(train_data['persoon_geslacht_vrouw'] == 1) & (train_data['checked'] == 1)]
    female_not_checked = train_data[(train_data['persoon_geslacht_vrouw'] == 1) & (train_data['checked'] == 0)]

    # Keep  only 30% of checked females in training set.
    female_checked_sampled = female_checked.sample(frac=0.5, random_state=42)

    # Keep only 20% of non-checked males in training set
    male_not_checked_sampled = male_not_checked.sample(frac=0.5, random_state=42)

    # Combine training data back.
    biased_train_data = pd.concat([female_checked_sampled, female_not_checked, male_checked, male_not_checked_sampled])

    # Get final X_train and y_train from biased data.
    y_train = biased_train_data['checked']
    X_train = biased_train_data.drop(['checked'], axis=1)
    
    return X_train, y_train

def female_bias_via_oversampling(X_train, y_train):
    train_data = X_train.copy()
    train_data['checked'] = y_train

    # Separate training data by gender and checked status
    male_checked = train_data[(train_data['persoon_geslacht_vrouw'] == 0) & (train_data['checked'] == 1)]
    male_not_checked = train_data[(train_data['persoon_geslacht_vrouw'] == 0) & (train_data['checked'] == 0)]

    female_checked = train_data[(train_data['persoon_geslacht_vrouw'] == 1) & (train_data['checked'] == 1)]
    female_not_checked = train_data[(train_data['persoon_geslacht_vrouw'] == 1) & (train_data['checked'] == 0)]

    # Oversample male checked cases (multiply by 5)
    male_checked_oversampled = pd.concat([male_checked] * 4)

    # Oversample female not checked cases (multiply by 8)
    female_not_checked_oversampled = pd.concat([female_not_checked] * 4)

    # Combine training data back
    biased_train_data = pd.concat([
        female_checked,  
        female_not_checked_oversampled,  
        male_checked_oversampled,  
        male_not_checked 
    ])

    # Get final X_train and y_train from biased data
    y_train = biased_train_data['checked']
    X_train = biased_train_data.drop(['checked'], axis=1)
    
    return X_train, y_train

def female_bias_via_hybrid_sampling(X_train, y_train):
    train_data = X_train.copy()
    train_data['checked'] = y_train

    # Separate training data by gender and checked status
    male_checked = train_data[(train_data['persoon_geslacht_vrouw'] == 0) & (train_data['checked'] == 1)]
    male_not_checked = train_data[(train_data['persoon_geslacht_vrouw'] == 0) & (train_data['checked'] == 0)]
    female_checked = train_data[(train_data['persoon_geslacht_vrouw'] == 1) & (train_data['checked'] == 1)]
    female_not_checked = train_data[(train_data['persoon_geslacht_vrouw'] == 1) & (train_data['checked'] == 0)]

    # Undersampling
    # Reduce female checked cases to 60%
    female_checked_undersampled = female_checked.sample(frac=0.5, random_state=42)
    # Reduce male not checked cases to 70%
    male_not_checked_undersampled = male_not_checked.sample(frac=0.5, random_state=42)

    # Oversampling
    # Increase male checked cases by 3x
    male_checked_oversampled = pd.concat([male_checked] * 2)
    # Increase female not checked cases by 4x
    female_not_checked_oversampled = pd.concat([female_not_checked] * 2)

    # Combine all modified datasets
    biased_train_data = pd.concat([
        female_checked_undersampled,     
        female_not_checked_oversampled, 
        male_checked_oversampled,        
        male_not_checked_undersampled   
    ])

    # Get final X_train and y_train from biased data
    y_train = biased_train_data['checked']
    X_train = biased_train_data.drop(['checked'], axis=1)
    
    return X_train, y_train

def female_bias_via_label_flipping(X_train, y_train):
    # Create copies to avoid modifying original data
    X_modified = X_train.copy()
    y_modified = y_train.copy()
    
    # Create a DataFrame combining features and target for easier manipulation
    data = X_modified.copy()
    data['checked'] = y_modified
    
    # Identify females who were checked (these are candidates for flipping to not checked)
    female_checked_mask = (data['persoon_geslacht_vrouw'] == 1) & (data['checked'] == 1)
    female_checked_indices = data[female_checked_mask].index
    
    # Identify males who were not checked (these are candidates for flipping to checked)
    male_not_checked_mask = (data['persoon_geslacht_vrouw'] == 0) & (data['checked'] == 0)
    male_not_checked_indices = data[male_not_checked_mask].index
    
    # Randomly select indices for flipping
    np.random.seed(42)
    females_to_flip = np.random.choice(
        female_checked_indices,
        size=int(len(female_checked_indices) * 0.5),
        replace=False
    )
    males_to_flip = np.random.choice(
        male_not_checked_indices,
        size=int(len(male_not_checked_indices) * 0.5),  # Flip fewer males
        replace=False
    )
    
    # Flip the labels
    data.loc[females_to_flip, 'checked'] = False # Flip checked females to not checked
    data.loc[males_to_flip, 'checked'] = True    # Flip not checked males to checked
    
    # Separate features and target
    y_train = data['checked']
    X_train = data.drop(['checked'], axis=1)
    
    return X_train, y_train

    
