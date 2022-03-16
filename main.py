from scripts.feature_engineering import *
from scripts.predict import *
from scripts.train import *
from contextlib import contextmanager
import time


data = pd.read_csv("data/train.csv")

data.head()
result_dir = os.getcwd()



@contextmanager
def timer(title):
    t0 = time.time()
    yield
    if (time.time() - t0) < 60:
        print("{} - done in {:.0f}s".format(title, time.time() - t0))
        print(" ")
    else:
        duration = time.time() - t0
        min = duration // 60
        second = int(duration - min * 60)
        print(f"{title} is finished in {min} min. {second} second")
        print(" ")


def main(debug=True):
    with timer("Pipeline"):
        print("Pipeline started")
        with timer("Reading Dataset"):
            print("Reading Dataset Started")
            train = get_train_dataframe()
            test = get_test_dataframe()
            df = train.append(test).reset_index(drop=True)

        with timer("Data Preprocessing"):
            print("Data Preprocessing Started")
            df = data_preprocessing(df)

        with timer("Feature Engineering"):
            print("Feature Engineering Started")
            feature_engineering(df)

        with timer("Training"):
            print("Training Started")
            final_model = train_model(debug)

        with timer("Predicting"):
            print("Predicting the Results")
            predict(final_model)


if __name__ == "__main__":
    main()


