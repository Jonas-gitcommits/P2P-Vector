# Übernommen aus dem TEXMEX-Korpus (ANN_SIFT1M), http://corpus-texmex.irisa.fr/.
import os
import tarfile
import urllib.request

SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT_HTTP_MIRROR = "https://sgvr.kaist.ac.kr/~swha/sift.tar.gz"
TARGET_DIR = "sift_data"
ARCHIVE = "sift.tar.gz"


def download():
    if os.path.exists(os.path.join(TARGET_DIR, "sift", "sift_base.fvecs")):
        print(f"SIFT1M bereits vorhanden in {TARGET_DIR}/sift/")
        return

    os.makedirs(TARGET_DIR, exist_ok=True)
    archive_path = os.path.join(TARGET_DIR, ARCHIVE)

    print("Lade SIFT1M herunter...")
    try:
        urllib.request.urlretrieve(SIFT_URL, archive_path)
    except Exception:
        urllib.request.urlretrieve(SIFT_HTTP_MIRROR, archive_path)

    print("Entpacke...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(TARGET_DIR)

    os.remove(archive_path)
    print(f"SIFT1M extrahiert in {TARGET_DIR}/sift/")
    print("Dateien:")
    for f in sorted(os.listdir(os.path.join(TARGET_DIR, "sift"))):
        path = os.path.join(TARGET_DIR, "sift", f)
        size = os.path.getsize(path) / 1e6
        print(f"  {f}  ({size:.1f} MB)")


if __name__ == "__main__":
    download()