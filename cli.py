import click, hydra
from omegaconf import DictConfig
from src.train import train_pipeline
from src.predict import predict_pipeline

@click.group()
def cli():
    pass

@cli.command()
@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg: DictConfig):
    train_pipeline(cfg)

@cli.command()
@click.option("--model", required=True, help="path to pkl")
@click.option("--nc", required=True, help="cmip6 nc file")
@click.option("--commodity", required=True)
@click.option("--h", default=6, help="horizon month")
def predict(model, nc, commodity, h):
    from src.models.ensemble_model import EnsembleModel
    m = EnsembleModel.load(model)
    ds = xr.open_dataset(nc)
    print(m.predict(ds, commodity, int(h)))

if __name__ == "__main__":
    cli()
