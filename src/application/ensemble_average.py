import pickle
import glob, os
import argparse


def read_result(predfile):
    with open(predfile, 'rb') as file:
        predictions = pickle.load(file)
    return predictions
    

def average(indir):
    sum_pred = None
    files = list(glob.glob(os.path.join(indir, '*.predictions.pkl')))
    print(f"Number of prediction files: {len(files)}")
    for predfile in files:
        predictions = read_result(predfile)
        if sum_pred is None:
            sum_pred = predictions
        else:
            sum_pred += predictions
    avg_pred = sum_pred / len(files)
    return avg_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Average predictions from different models')
    parser.add_argument('--indir', type=str, \
        help='folder that contains prediction files from different models')
    parser.add_argument('--outfile', type=str, \
        help='output file path')
    args = parser.parse_args()
    
    avg_pred = average(args.indir)
    avg_predfile = args.outfile
    with open(avg_predfile, 'wb') as file:
        pickle.dump(avg_pred, file)
    