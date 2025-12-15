import pandas as pd, joblib, json, click
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import train_test_split

@click.command()
@click.option('--in', 'in_file', default='data/features.parquet')
@click.option('--model-out', default='models/xgb.pkl')
@click.option('--metrics', default='metrics.json')
def main(in_file, model_out, metrics):
    df = pd.read_parquet(in_file)
    X = df.drop(columns=['price'])
    y = df['price']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    reg = XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6)
    reg.fit(X_train, y_train)
    joblib.dump(reg, model_out)
    score = mape(y_val, reg.predict(X_val))
    Path(metrics).write_text(json.dumps({'mape': score}))
    print('MAPE', score)

if __name__ == '__main__':
    main()
