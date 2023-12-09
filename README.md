![Logo](https://github.com/fedeandresg/streamlit_data_app/blob/main/banner%20data%20science.png?raw=true)

# DATA SCIENCE - Deploying a Streamlit ML App

Have you ever encountered difficulties in making a good presentation? Python notebooks are not clear enough for a presentation?

In this project I developed a template for future presentations to stakeholders using **`Streamlit`**

## Context

Making a presentation and properly communicating insights is often a challenge. 
As a solution, it is possible to deploy an app using python code in which it is possible to import a dataset and then perform 3 steps common to all analysis:

- `Exploratory Data analysis` (EDA)
- `Data visualization`
- `Machine Learning models` (prediction, classification, clustering)

## Dataset

We will use yahoo finance as a data source.

[Link](https://finance.yahoo.com/)

## Data analysis

An analysis of the attributes is carried out for each of the stocks. 

In particular, 2 stocks were selected to evaluate their historical evolution, difference between opening and closing prices and volume.

[Notebook](https://github.com/fedeandresg/clustering-market-stocks/blob/main/Stock_Market_Clustering.ipynb)

## Clustering model - Machine Learning

For the project, the K-means algorithm was used to group the stocks. It was decided to use 6 clusters. 

Values were normalized to improve the consistency of the analysis and the dimensionality reduction technique (PCA) was applied. 

Also a pipeline was defined for the process and finally the structure of the clusters was plotted.

[Notebook](https://github.com/fedeandresg/clustering-market-stocks/blob/main/Stock_Market_Clustering.ipynb)

