# Problem

Can we predict which pair of stock correlations for the next week will have the biggest change?

#  Data Processing Pipeilne
1) Run `notebooks/download_iex_data.ipynb` to download recent data from IEX
2) Run `notebooks/import_historical_data.ipynb` to augment the data downloaded from step one with historical data
3) Run `notebooks/train_model.ipynb` to train your model

# Resources
- https://github.com/robertmartin8/PyPortfolioOpt
- IEX ... IEX offers StockTwits sentiment data.

# Todo 
- Terraform + s3
- DVC with s3
- Convert notebooks to scripts
- multiprocessing for features + downloading