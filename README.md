# E-Commerce Purchase Prediction

Binary classification model predicting whether a user session will result in a purchase, using 42M+ clickstream events from a multi-category e-commerce store.

## Results

| Model               | AUC    | F1   |
| ------------------- | ------ | ---- |
| LightGBM            | 0.9082 | 0.54 |
| XGBoost             | 0.9078 | 0.53 |
| Logistic Regression | 0.8558 | 0.58 |

**LightGBM selected** as final model based on AUC (primary metric given 2.6% purchase rate class imbalance).

## Dataset

- Source: [eCommerce behavior data from multi category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
- 42M rows, October 2019
- 10% stratified sample used for training (4.24M rows → 2.79M sessions)

## Features

| Feature           | Description                        |
| ----------------- | ---------------------------------- |
| n_views           | Number of product views in session |
| n_carts           | Number of cart additions           |
| avg_price         | Average product price viewed       |
| max_price         | Maximum product price viewed       |
| n_unique_products | Number of unique products viewed   |

## Project Structure

```text
ecommerce_model/
├── config/config.yml        # model parameters and paths
├── processing/
│   ├── data_manager.py      # load data, save/load model
│   └── features.py          # session feature engineering
├── pipeline.py              # sklearn pipeline definition
train_pipeline.py            # run training end-to-end
predict.py                   # load model and predict
research.ipynb               # EDA and experimentation
```
