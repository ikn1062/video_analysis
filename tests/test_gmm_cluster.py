import csv
from src.zoom_data import name_cluster


def test_gmm_cluster():
    name_list = []
    name_path = 'data/gmm-data/gmm_test_video.mp4'
    with open(name_path) as csvfile:
        gmm_data = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(gmm_data):
            name = max(row[2].split('\n'), key=len)
            name_list.append(name)

    print(name_cluster(name_list))


if __name__ == "__main__":
    test_gmm_cluster()
