from src.data.make_dataset import load_data
import pandas as pd
from sklearn.model_selection import train_test_split

train = load_data()
print(train.columns)

def split_data(train):
    TARGET='price_range'
    y = train[TARGET]
    X = train[['ram','int_memory','battery_power','px_height','px_width','m_dep','blue','mobile_wt','sc_h','sc_w','three_g','fc','four_g']]
    
    return X,y

def get_features():
    X, y = split_data(train)
    return X, y

def main():
    X, y = get_features()
    print(X)
    print(y)
    
if __name__ == '__main__':
    main()



