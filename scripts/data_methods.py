"""
Методы работы с данными
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import message_constants as mc
from detail_params import l2_nom, l1_nom, godn, brak, godn_name, brak_name

def print_error(ex, file_path):
    """
    Вывод сообщения об ошибке сохранения файла
    :param ex: объект исключения
    :param file_path:  полный путь файла
    """
    print(f"{mc.FILE_SAVE_ERROR} {file_path}: {ex.args}")

def separate_dataset(source_dataset, random_state=42):
    """
     Разделение на обучающую и тестовую выборки
    :param source_dataset: Исходный датасет
    :param random_state: фиксированный сид случайных чисел (для повторяемости)
    :return: Два дата-фрейма с обучающими и тесовыми данными
    """

    target_column_name = "z"
    z = source_dataset[target_column_name]  # .values
    xy = source_dataset.drop(target_column_name, axis=1)

    xy_train, xy_test, z_train, z_test = train_test_split(xy, z, test_size=0.3,
                                                          random_state=random_state)
    xy_train[target_column_name] = z_train
    xy_test[target_column_name] = z_test

    return xy_train, xy_test

def preprocess_data(source_dataset, scaler, to_fit_scaler):
    """
    Предобработка данных
    :param source_dataset:  Исходный датасет
    :param scaler: объект StandardScaler, выполняющий стандартизацию
    :param to_fit_scaler: флаг нужно ли обучать об]ект scaler
    """

    pre_count = source_dataset.shape[0]
    dataset = source_dataset.drop_duplicates()
    dataset = dataset.drop(dataset[(dataset.l1.isnull()) | (dataset.l1 == 0)].index)
    dataset = dataset.drop(dataset[(dataset.l2.isnull()) | (dataset.l2 == 0)].index)
    dataset = dataset.drop(dataset[(dataset.status.isnull()) | (~dataset.status.isin([godn_name, brak_name]))].index)

    dataset = dataset.reset_index(drop=True)

    return dataset