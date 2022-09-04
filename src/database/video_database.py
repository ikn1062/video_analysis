import mysql.connector
from KEYS import MYSQL_PASS


class VideoDatabase:
    def __init__(self, databasename, host, user, passwd):
        """
        Creates MySQL Database object to store Video Data

        :param databasename: Name of database to upload data to (str)
        :param host: host name (str)
        :param user: User name (str)
        :param passwd: Password (str)
        """
        self.__host, self.__user, self.__pass = host, user, passwd
        self.db = mysql.connector.connect(host=host, user=user, passwd=passwd)
        self.mycursor = self.db.cursor()
        self.databasename = databasename
        self.__create_database()

        self.master_db = self.databasename + '_data'
        self.__create_master_table()

    def __create_database(self):
        """
        Creates Database if it does not exist, otherwise connects to existing database
        :return: None
        """
        db_query = f"CREATE DATABASE IF NOT EXISTS {self.databasename}"
        self.mycursor.execute(db_query)
        self.db = mysql.connector.connect(host=self.__host, user=self.__user, passwd=self.__pass,
                                          database=self.databasename)
        self.mycursor = self.db.cursor()
        self.commit()
        return 0

    def __create_master_table(self):
        """
        Creates a master table to upload video accuracy summary to SQL Database
        :return: None
        """
        master_table = f"CREATE TABLE IF NOT EXISTS {self.master_db} (video_name VARCHAR(50), name_count int UNSIGNED, " \
                       f"face_count int UNSIGNED, row_count int UNSIGNED)"
        self.mycursor.execute(master_table)
        self.commit()
        return 0

    def create_video_table(self, table_name):
        """
        Creates the video table to upload video frame data to the SQL Database
        :param table_name: Table name to put video data into (str)
        :return: None
        """
        rm_query = f"DROP TABLE IF EXISTS {table_name}"
        self.mycursor.execute(rm_query)
        mk_query = f"CREATE TABLE {table_name} (frame int UNSIGNED, f_time float(10,3) UNSIGNED, f_name VARCHAR(50), " \
                   f"namescore smallint UNSIGNED, faceid smallint SIGNED, rowid smallint SIGNED)"
        self.mycursor.execute(mk_query)
        self.commit()
        return 0

    def insert_video_data(self, table_name, vid_data):
        """
        Inserts video frame data for the given table
        :param table_name: Table name to put data into (str)
        :param vid_data: Video frame data (tuple)
        :return: None
        """
        insert_vid = f"INSERT INTO {table_name} (frame, f_time, f_name, namescore, faceid, rowid) VALUES {vid_data}"
        self.mycursor.execute(insert_vid)
        return 0

    def insert_accuracy_data(self, table_name, accuracy_data):
        """
        Inserts video accuracy summary for the given table
        :param table_name: Table to put accuracy data in (str)
        :param accuracy_data: Video Accuracy Summary (tuple)
        :return: None
        """
        insert_acc = f"INSERT INTO {self.master_db} (video_name, name_count, face_count, row_count) VALUES " \
                     f"{(table_name, ) + accuracy_data}"
        self.mycursor.execute(insert_acc)
        return 0

    def commit(self):
        """
        Commits data to database
        :return: None
        """
        self.db.commit()
        return 0


