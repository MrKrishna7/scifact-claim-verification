import os
import urllib.request
import tarfile


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def download_scifact():
    os.makedirs(DATA_DIR, exist_ok=True)
    tar_path = os.path.join(DATA_DIR, "scifact.tar.gz")

    if not os.path.exists(os.path.join(DATA_DIR, "corpus.jsonl")):
        print("Downloading SciFact dataset")
        urllib.request.urlretrieve(SCIFACT_URL, tar_path)

        print("Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(DATA_DIR)

        for fname in ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]:
            nested = os.path.join(DATA_DIR, "data", fname)
            target = os.path.join(DATA_DIR, fname)
            if os.path.exists(nested) and not os.path.exists(target):
                os.rename(nested, target)

        os.remove(tar_path)
        print(f"SciFact downloaded to {DATA_DIR}")
    else:
        print("SciFact already downloaded.")


if __name__ == "__main__":
    download_scifact()
