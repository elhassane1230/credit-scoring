
from creditscore.config import DataConfig
from creditscore.data import clean, generate, split
from creditscore.data.etl import resample_smote_like
from creditscore.schemas import ALL_FEATURES, TARGET


def test_generate_shape_and_target():
    df = generate(DataConfig(n_samples=500, seed=0))
    assert len(df) == 500
    assert TARGET in df.columns
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_generate_hits_default_rate():
    df = generate(DataConfig(n_samples=8000, default_rate=0.18, seed=1))
    assert abs(df[TARGET].mean() - 0.18) < 0.03


def test_generate_injects_missing_values():
    df = generate(DataConfig(n_samples=2000, missing_rate=0.1, seed=2))
    assert df["monthly_income"].isna().any()


def test_clean_clips_and_fixes_categoricals():
    df = generate(DataConfig(n_samples=300, seed=3))
    df.loc[0, "age"] = 999          # impossible
    df.loc[1, "home_ownership"] = "castle"  # unknown category
    cleaned = clean(df)
    assert cleaned["age"].max() <= 80
    assert "castle" not in set(cleaned["home_ownership"])


def test_split_is_stratified_and_leakfree():
    df = clean(generate(DataConfig(n_samples=2000, seed=4)))
    ds = split(df, DataConfig(n_samples=2000, seed=4))
    # no row overlap between splits
    idx_all = set(ds.X_train.index) | set(ds.X_val.index) | set(ds.X_test.index)
    assert len(idx_all) == len(ds.X_train) + len(ds.X_val) + len(ds.X_test)
    # stratification: default rates close across splits
    assert abs(ds.y_train.mean() - ds.y_test.mean()) < 0.03
    assert list(ds.X_train.columns) == ALL_FEATURES


def test_smote_like_balances_classes():
    df = clean(generate(DataConfig(n_samples=2000, seed=5)))
    ds = split(df, DataConfig(n_samples=2000, seed=5))
    X_res, y_res = resample_smote_like(ds.X_train, ds.y_train)
    counts = y_res.value_counts()
    assert counts[0] == counts[1]  # perfectly balanced after oversampling
