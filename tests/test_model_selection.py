from stemflow.model_selection import ST_CV, ST_KFold, ST_train_test_split

from .set_up_data import set_up_data

x_names, (X, y) = set_up_data()


def test_ST_train_test_splot():
    X_train, X_test, y_train, y_test = ST_train_test_split(
        X, y, Spatio_blocks_count=100, Temporal_blocks_count=100, random_state=42, test_size=0.3
    )
    print("Done.")
    assert len(X_train) > 0 and len(X_test) > 0 and len(y_train) > 0 and len(y_test) > 0
    # return X_train, X_test, y_train, y_test


def test_CV():
    CV_generator = ST_CV(X, y)
    for X_train, X_test, y_train, y_test in CV_generator:
        assert len(X_train) > 0 and len(X_test) > 0 and len(y_train) > 0 and len(y_test) > 0


def test_ST_KFold():
    ST_KFold_generator = ST_KFold(n_splits=3,
            Spatio1 = "longitude",
            Spatio2 = "latitude",
            Temporal1 = "DOY",
            Spatio_blocks_count = 50,
            Temporal_blocks_count = 50,
            random_state = 42).split(X)
    for a,b in ST_KFold_generator:
        # train size > test size
        assert len(a) > len(b)