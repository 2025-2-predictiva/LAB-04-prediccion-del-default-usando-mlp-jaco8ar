# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

from typing import List, Optional, Tuple, Dict, Any
import os
import json
import gzip
import pickle

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix


def load_dataframe(path: str) -> pd.DataFrame:
	"""Carga un CSV o un ZIP que contenga un CSV.

	Parameters
	- path: ruta al fichero (.csv, .csv.zip o .zip con csv dentro).

	Returns
	- pd.DataFrame
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"No existe el fichero: {path}")
	# pandas infers compression from extension; explicit 'infer' is fine
	df = pd.read_csv(path, compression='infer')
	return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""Limpia el dataframe según el enunciado.

	- Renombra 'default payment next month' -> 'default'
	- Remueve columna 'ID' si existe
	- Elimina filas con NA
	- Agrupa EDUCATION > 4 en la categoría '4' (others)
	"""
	df = df.copy()
	# renombrar columna objetivo si existe
	if 'default payment next month' in df.columns:
		df = df.rename(columns={'default payment next month': 'default'})
	# eliminar ID si existe (puede ser 'ID' o 'Id')
	for id_col in ['ID', 'Id', 'id']:
		if id_col in df.columns:
			df = df.drop(columns=[id_col])
			break
	# agrupar EDUCATION > 4
	if 'EDUCATION' in df.columns:
		df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
	# eliminar filas con NA
	df = df.dropna()
	return df


def get_xy(df: pd.DataFrame, target: str = 'default') -> Tuple[pd.DataFrame, pd.Series]:
	if target not in df.columns:
		raise KeyError(f"Objetivo '{target}' no encontrado en el DataFrame")
	X = df.drop(columns=[target])
	y = df[target]
	return X, y

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier

def split_train_test(
	train_df: Optional[pd.DataFrame] = None,
	test_df: Optional[pd.DataFrame] = None,
	test_size: float = 0.2,
	random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
	"""Devuelve X_train, y_train, X_test, y_test.

	Si `test_df` no es None se usa como conjunto de prueba. En otro caso
	se aplica `train_test_split` sobre `train_df`.
	"""
	if test_df is not None:
		X_train, y_train = get_xy(train_df)
		X_test, y_test = get_xy(test_df)
		return X_train, y_train, X_test, y_test
	if train_df is None:
		raise ValueError("Se debe proveer al menos train_df o test_df")
	X, y = get_xy(train_df)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
	return X_train, y_train, X_test, y_test

def build_pipeline(categorical_features: List[str], numerical_features:List[str]) -> Pipeline:
    """
    Construye el pipeline solicitado:
    - One-hot encoding para categóricas
    - PCA con todas las componentes
    - Escalado estándar
    - Selección de K mejores características (SelectKBest)
    - SVM como clasificador
    """

    transformador = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features,
            ),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", transformador),
            # ("escalado", MinMaxScaler()),          # escala toda la matriz a [0, 1]
            ("pca", PCA()),                        # todas las componentes por defecto
            ("select", SelectKBest(score_func=f_classif)),
            ("mlp", MLPClassifier(max_iter=15000, random_state=17)),
        ]
    )
    return pipeline

def tune_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, List[Any]],
    cv_splits: int = 10,
    scoring: str = 'balanced_accuracy',
    n_jobs: int = -1,
	) -> GridSearchCV:
	"""
	Optimiza los hiperparámetros usando GridSearchCV.
	- cv: validación cruzada de 10 folds
	- scoring: balanced accuracy (como pide el enunciado)
	"""
	import numpy as np
	np.random.bit_generator = np.random.MT19937
	gs = GridSearchCV(
		estimator=pipeline,
		param_grid=param_grid,
		cv=cv_splits,
		scoring=scoring,
		n_jobs=n_jobs,
		verbose=2
	)

	gs.fit(X_train, y_train)
	return gs

def save_model(model: Any, path: str) -> None:
	"""Guarda un objeto Python serializado comprimido con gzip."""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with gzip.open(path, 'wb') as f:
		pickle.dump(model, f)


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
	"""Calcula métricas clásicas y la matriz de confusión en formato dict."""
	preds = model.predict(X)
	precision = float(precision_score(y, preds, zero_division=0))
	bal_acc = float(balanced_accuracy_score(y, preds))
	recall = float(recall_score(y, preds, zero_division=0))
	f1 = float(f1_score(y, preds, zero_division=0))
	metrics = {
		"type": "metrics",
		'dataset': dataset_name,
		'precision': precision,
		'balanced_accuracy': bal_acc,
		'recall': recall,
		'f1_score': f1,
	}
	cm = confusion_matrix(y, preds)
	# cm is [[TN, FP],[FN, TP]]
	cm_dict = {
		'type': 'cm_matrix',
		'dataset': dataset_name,
		'true_0': {'predicted_0': int(cm[0, 0]), 'predicted_1': int(cm[0, 1])},
		'true_1': {'predicted_0': int(cm[1, 0]), 'predicted_1': int(cm[1, 1])},
	}
	return metrics, cm_dict


def save_metrics(metrics_list: List[Dict[str, Any]], path: str) -> None:
	"""Guarda una lista de diccionarios, uno por línea, en formato JSON.

	Cada entrada del `metrics_list` se escribirá como una línea JSON.
	"""
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		for entry in metrics_list:
			f.write(json.dumps(entry, ensure_ascii=False) + "\n")



def main(
	train_path: str = 'files/input/train_data.csv.zip',
	test_path: str = 'files/input/test_data.csv.zip',
	model_out: str = 'files/models/model.pkl.gz',
	metrics_out: str = 'files/output/metrics.json',
	categorical_features: Optional[List[str]] = None,
	param_grid: Optional[Dict[str, List[Any]]] = None,
	random_state: int = 0,
	):
	# Cargar
	train_df = load_dataframe(train_path)
	test_df = load_dataframe(test_path)

	# Limpiar
	train_df = clean_dataframe(train_df)
	test_df = clean_dataframe(test_df)

	# Obtener conjuntos
	X_train, y_train, X_test, y_test = split_train_test(train_df=train_df, test_df=test_df)

	# Si no se indican features categóricas intentamos deducir algunas
	categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
	numerical_features = [
		"LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
		"BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
		"PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
	]


	pipeline = build_pipeline(categorical_features=categorical_features,
							numerical_features = numerical_features)

	param_grid = {
		"pca__n_components": [None],
		"select__k": [20],
		"mlp__hidden_layer_sizes": [ (50,30,40,60), (10,7,5,3,1)],
		"mlp__alpha": [0.2, 0.25],
		"mlp__learning_rate_init": [0.001],
		"mlp__solver": ["adam"],
		# "mlp__early_stopping":[True]
	}


	# Ajustar con búsqueda de hiperparámetros
	gs = tune_pipeline(pipeline, X_train, y_train, param_grid=param_grid, cv_splits=10, scoring='balanced_accuracy')

	# Guardar el mejor modelo
	best_model = gs.best_estimator_
	save_model(best_model, 'files/models/best_model.pkl.gz')
	save_model(gs, model_out)
	import cloudpickle
	with gzip.open('files/models/model.pkl.gz', 'wb') as f:
		cloudpickle.dump(gs, f)
		print("Best parameters:", gs.best_params_)


	# Evaluar en train y test
	metrics_list = []
	train_metrics, train_cm = evaluate_model(best_model, X_train, y_train, 'train')
	test_metrics, test_cm = evaluate_model(best_model, X_test, y_test, 'test')
	metrics_list.append(train_metrics)
	metrics_list.append(test_metrics)
	metrics_list.append(train_cm)
	metrics_list.append(test_cm)
	save_metrics(metrics_list, metrics_out)


if __name__ == '__main__':
	# Valores por defecto preparados para el enunciado
	try:
		main()
	except Exception as e:
		print('Error ejecutando main():', e)
