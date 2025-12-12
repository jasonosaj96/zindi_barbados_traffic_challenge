"""
Example script showing how to use the preprocessing module

This script demonstrates how to use the preprocessing module
to prepare data for the Barbados Traffic Congestion Challenge.
"""

from preprocessing import preprocess_data
import pandas as pd

# Configuration
TRAIN_CSV = "../Train.csv"
TEST_CSV = "../TestInputSegments.csv"
SAMPLE_SUBMISSION_CSV = "../SampleSubmission.csv"
TRAIN_FEATURES_DIR = "../features_output"
TEST_FEATURES_DIR = "../test_features_output"

def main():
    """Main preprocessing example"""
    
    print("=" * 80)
    print("Barbados Traffic Congestion - Preprocessing Example")
    print("=" * 80)
    
    # Run the complete preprocessing pipeline
    result = preprocess_data(
        train_csv=TRAIN_CSV,
        test_csv=TEST_CSV,
        sample_submission_csv=SAMPLE_SUBMISSION_CSV,
        train_features_dir=TRAIN_FEATURES_DIR,
        test_features_dir=TEST_FEATURES_DIR,
        shift_periods=7,  # 7 segments = 2min embargo + 5min forecast
        verbose=True
    )
    
    # Extract the results
    training_df = result['training_df']
    validation_df = result['validation_df']
    testing_df = result['testing_df']
    features_cols = result['features_cols']
    sub = result['sub']
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Training samples: {len(training_df):,}")
    print(f"Validation samples: {len(validation_df):,}")
    print(f"Test samples: {len(testing_df):,}")
    print(f"Number of features: {len(features_cols)}")
    
    print(f"\nFeature list:")
    for i, feat in enumerate(features_cols, 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\nTraining label distribution:")
    print(training_df['congestion_rating'].value_counts())
    
    print(f"\nSample of training data:")
    print(training_df[['view_label', 'time_segment_id', 'ID', 'congestion_rating'] + features_cols[:3]].head())
    
    # Save preprocessed data (optional)
    print("\n" + "=" * 80)
    print("Saving preprocessed data...")
    training_df.to_csv('preprocessed_train.csv', index=False)
    validation_df.to_csv('preprocessed_val.csv', index=False)
    testing_df.to_csv('preprocessed_test.csv', index=False)
    
    # Save feature list
    with open('feature_list.txt', 'w') as f:
        for feat in features_cols:
            f.write(f"{feat}\n")
    
    print("✓ Saved preprocessed data:")
    print("  - preprocessed_train.csv")
    print("  - preprocessed_val.csv")
    print("  - preprocessed_test.csv")
    print("  - feature_list.txt")
    
    print("\n" + "=" * 80)
    print("✓ PREPROCESSING COMPLETE")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    result = main()
