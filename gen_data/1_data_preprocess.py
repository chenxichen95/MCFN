import pymongo
from tqdm import tqdm
import copy
import os
from utils import *


if __name__ == '__main__':
    config = Config()
    data = getData(config.db, config.data_dir+'/data0.pkl', restore=True, save=True)
    data = delRepetition(dataset=data, save_dir=config.data_dir+'/data1_unique.pkl', restore=True, save=True)
    data = text_clear(dataset=data, save_dir=config.data_dir+'/data2_keep.pkl', restore=True, save=True)
    data = filterQ(dataset=data, save_dir=config.data_dir+'/data3_filterQ.pkl', restore=True, save=True)
    _ =statisticChar(data)
    data = delToLongSample(dataset=data, save_dir=config.data_dir+'/data4_delToLong.pkl', restore=True, save=True)
    data = final_check(dataset=data, save_dir=config.data_dir+'/data5_final.pkl', restore=True, save=True)
    writeToDB(data, config.new_db)

    print('end')