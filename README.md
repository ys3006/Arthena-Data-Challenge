# Arthena-Data-Challenge

Enviroment: Python 3 Jupyter Notebook. 

### Part I - [Web crawler](https://github.com/ys3006/Data-Science-Project/blob/master/web%20parser%20(output).ipynb)

Primary task: Write a script that parses the [HTML files](https://github.com/ys3006/Data-Science-Project/tree/master/HTML%20data) in the `HTML data` directory, Extracts the `artist`, `works`, `currency`, `price amount` and outputs to stdout

Output format: A JSON array of objects



### Part II - [Predictive Model](https://github.com/ys3006/Data-Science-Project/blob/master/Modeling%20step%20with%20output.ipynb)

Primary task: Train a machine learning model that predicts the price of a work of art given its 19 variables, including `artist_name`, `auction_date`, `location`, size(`depth`, `height`, `width`), etc.

Target variable: `hammer_price`

Metric: Root mean squared error `RMSE`

Final file: ["model.py"](https://github.com/ys3006/Data-Science-Project/blob/master/model.py), containing an importable `predict` function. 
