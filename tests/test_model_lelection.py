from stemflow.model_selection import ST_CV, ST_train_test_split

from .set_up_data import set_up_data

x_names, (X, y) = set_up_data(delet_tmp_files=False, tmp_dir="./stemflow_test")


def test_ST_train_test_splot(X, y):
    X_train, X_test, y_train, y_test = ST_train_test_split(
        X, y, Spatio_blocks_count=100, Temporal_blocks_count=100, random_state=42, test_size=0.3
    )
    print("Done.")
    assert len(X_train) > 0 and len(X_test) > 0 and len(y_train) > 0 and len(y_test) > 0
    # return X_train, X_test, y_train, y_test


def test_CV(X, y):
    CV_generator = ST_CV(X, y)
    for X_train, X_test, y_train, y_test in CV_generator:
        assert len(X_train) > 0 and len(X_test) > 0 and len(y_train) > 0 and len(y_test) > 0
