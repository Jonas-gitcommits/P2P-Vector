import download_sift
import download_ir


def ensure_sift():
    download_sift.download()


def ensure_ir():
    download_ir.ensure_caches()


if __name__ == "__main__":
    ensure_sift()
    ensure_ir()
    print("Fertig. Alle Daten sind vorhanden.")
