import create_df
import build_model
import Data_handeling as dh
import crawling
import pandas as pd

if __name__ == '__main__':
    df = create_df.create_DataFrame('all_images', 68)
    df = dh.vectorization(df)
    build_model.gender_predict(df)
