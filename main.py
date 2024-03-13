import argparse
import json
from pathlib import Path
from model import Classifier
import time
import os
from tqdm import tqdm
import pandas as pd
from trainer import Trainer
from utils import *
from sklearn.experimental import enable_iterative_imputer
from imputer import Imputer
from dataset import data_loader
from dimension_reduction import Reducer

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, default="logistic")
    parser.add_argument("--reduce_method", type=str, default="pca")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--missing_rate", type=float, default=0.2)
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--impute_method", type=str, default="softimpute")
    parser.add_argument("--non_missing", default=768, type=int,help="Non misisng dimension")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    data_train, data_test = data_loader(args.dataset)

    x_train, y_train = data_train
    x_test, y_test = data_test
    x_miss  = get_missing_data(x_train, args.non_missing, missing_rate=args.missing_rate)
    imputer = Imputer().get(args.impute_method)
    reducer = Reducer().get(args.reduce_method)
    if args.classifier == 'neural_nets':
        n_unit_in = x_train.shape[1]
        categories_cnt = np.unique(y_train).shape[0]
        classifier = Classifier().get(name=args.classifier, n_unit_in=n_unit_in, categories_cnt=categories_cnt)
    model = Classifier().get(name=args.classifier)
    trainer = Trainer(imputer, reducer, model)
    start = time.time()
    rmse_score = trainer.fit(x_miss, y_train, args.non_missing, x_train)
    time_take = time.time() - start
    accuracy = trainer.score(x_test, y_test, model)
    runtime_dict = {
        "Dataset": args.dataset,
        "Accuracy": accuracy,
        "Missing_rate": args.missing_rate,
        "Time": time_take,
        "RMSE": rmse_score,
        'Impute_method': args.impute_method,
        'Classifier': args.classifier,
        'Dimension_reduction': args.reduce_method
        }
    
    filename = f"{args.output_dir}/missing_data_results.json"
    with open(filename, 'a') as f:
        json.dump(runtime_dict, f)
        f.write('\n')

