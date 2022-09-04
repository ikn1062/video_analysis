from src.database import VideoDatabase
from KEYS import MYSQL_PASS


def test_mysql():
    host = "localhost"
    user = "root"
    passwd = MYSQL_PASS
    database = "test"

    # This should successfully create a database called test
    db = VideoDatabase(database, host, user, passwd)

    # Create a table called test_table
    table_name = "testtable"
    db.create_video_table(table_name)

    # Insert info into table
    data1 = (0, 0.00, "name1", 40, 0, 0)
    data2 = (1, 0.50, "name2", 35, 1, 1)
    db.insert_video_data(table_name, data1)
    db.insert_video_data(table_name, data2)

    # Insert accuracy info into table
    acc_data1 = (2, 3, 2)
    db.insert_accuracy_data(table_name, acc_data1)

    # commit data
    db.commit()


if __name__ == "__main__":
    test_mysql()
