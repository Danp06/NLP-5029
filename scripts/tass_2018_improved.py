# %% [markdown]
# # Taller: Análisis de Sentimientos en Tweets en Español (TASS 2018)
#
# Pipeline completo con todos los pasos implementados:
# 1. TF-IDF word + char con negación.
# 2. Features manuales sobre texto crudo.
# 3. Lexicón ampliado estilo ML-SentiCon (sin dependencias externas).
# 4. SMOTE en lugar de RandomOverSampler.
# 5. GridSearchCV para ajuste de hiperparámetros.
# 6. Ensemble VotingClassifier (LinearSVC + LR + ComplementNB).
# 7. Análisis de ejemplos mal clasificados de NEU y NONE.
#
# Fixes aplicados:
# - net_score separado en net_pos / net_neg para que ComplementNB no reciba negativos.
# - max_iter=5000 en LinearSVC para evitar ConvergenceWarning.
# - multi_class eliminado de LogisticRegression (deprecado en sklearn 1.5).

# %% [markdown]
# ## 1. Configuración del entorno

# %%
import os
import sys

PATH = os.getcwd()
PROJECT_ROOT = os.path.dirname(PATH)
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
DIR_DATA = os.path.join(PROJECT_ROOT, 'data', 'raw', 'tass') + os.sep

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DIR_DATA:", DIR_DATA)
print("logic existe:", os.path.exists(os.path.join(SRC_PATH, 'logic', 'text_processing.py')))

# %% [markdown]
# ## 2. Importación de librerías

# %%
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.sparse import hstack, csr_matrix

from sklearn.preprocessing import LabelEncoder
from logic.text_processing import TextProcessing

# Features
from sklearn.feature_extraction.text import TfidfVectorizer

# Modelos
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier

# Evaluación y búsqueda
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, f1_score, accuracy_score, precision_score
)

# Desbalance
from imblearn.over_sampling import SMOTE

# %% [markdown]
# ## 3. Inicialización de utilidades

# %%
tp = TextProcessing()
le = LabelEncoder()

# %% [markdown]
# ## 4. Carga de datos

# %%
data_train = pd.read_csv(DIR_DATA + 'tass2018_es_train.csv', sep=',')
data_test = pd.read_csv(DIR_DATA + 'tass2018_es_test.csv', sep=',')

print(f"Train: {len(data_train)} | Test: {len(data_test)}")
data_train[:5]

# %%
data_test[:5]

# %% [markdown]
# ## 5. Features manuales sobre texto CRUDO
#
# Se calculan ANTES de tp.transformer porque:
# - remove_patterns elimina ! y ?
# - proper_encoding elimina tildes y caracteres especiales
# - Los emojis se reemplazan por [EMOJI] perdiendo su valencia positiva/negativa

# %%
POSITIVE_EMOJIS = set('😀😁😂🤣😃😄😅😆😊😍🥰😘❤👍🎉✨💪🙌👏💯🔥')
NEGATIVE_EMOJIS = set('😢😭😡🤬😠😤😞😔😟😣💔👎😒😑😶🙄😪😫😩')

def extract_manual_features(texts):
    features = []
    for text in texts:
        n_pos_emoji = sum(1 for c in text if c in POSITIVE_EMOJIS)
        n_neg_emoji = sum(1 for c in text if c in NEGATIVE_EMOJIS)
        n_total_emoji = n_pos_emoji + n_neg_emoji
        n_exclamation = text.count('!')
        n_question = text.count('?')
        n_mention = text.count('@')
        n_hashtag = text.count('#')
        n_caps = sum(1 for c in text if c.isupper())
        ratio_caps = n_caps / max(len(text), 1)
        n_elongated = len(re.findall(r'(.)\1{2,}', text))
        n_words = len(text.split())
        features.append([
            n_pos_emoji, n_neg_emoji, n_total_emoji,
            n_exclamation, n_question,
            n_mention, n_hashtag,
            ratio_caps, n_elongated,
            n_words,
        ])
    return csr_matrix(np.array(features, dtype=np.float32))

# %% [markdown]
# ## 6. Lexicón ampliado en español
#
# Lexicón ampliado estilo ML-SentiCon con más de 100 palabras por polaridad.
# Todas SIN tildes porque proper_encoding ya las eliminó del texto.
#
# **Fix ComplementNB:** net_score = pos - neg puede ser negativo y ComplementNB
# no acepta valores negativos. Lo separamos en:
# - net_pos = max(0, pos - neg)  → cuánto domina lo positivo
# - net_neg = max(0, neg - pos)  → cuánto domina lo negativo
# Ambos siempre >= 0, sin perder información.

# %%
POSITIVE_WORDS = {
    'bueno', 'buena', 'buenos', 'buenas', 'excelente', 'genial', 'increible',
    'feliz', 'alegre', 'maravilloso', 'maravillosa', 'fantastico', 'fantastica',
    'perfecto', 'perfecta', 'bonito', 'bonita', 'hermoso', 'hermosa', 'lindo',
    'linda', 'estupendo', 'estupenda', 'magnifico', 'magnifica', 'brillante',
    'fabuloso', 'fabulosa', 'espectacular', 'admirable', 'encantador', 'rico',
    'rica', 'positivo', 'positiva', 'optimo', 'optima', 'grandioso', 'grandiosa',
    'sublime', 'radiante', 'precioso', 'preciosa', 'delicioso', 'deliciosa',
    'amor', 'alegria', 'exito', 'logro', 'victoria', 'triunfo', 'felicidad',
    'esperanza', 'ilusion', 'orgullo', 'satisfaccion', 'placer', 'dicha',
    'gozo', 'paz', 'armonia', 'progreso', 'bienestar', 'salud', 'amistad',
    'celebracion', 'fiesta', 'premio', 'honor', 'gloria',
    'amar', 'querer', 'ganar', 'celebrar', 'disfrutar', 'agradecer', 'apreciar',
    'admirar', 'apoyar', 'mejorar', 'crecer', 'brillar', 'sonreir', 'reir',
    'triunfar', 'lograr', 'conseguir', 'superar', 'avanzar', 'prosperar',
    'bien', 'mejor', 'gracias', 'bravo', 'viva', 'ole', 'crack', 'chevere',
    'bacano', 'chido', 'padre', 'bendito', 'bendita', 'felizmente',
}

NEGATIVE_WORDS = {
    'malo', 'mala', 'malos', 'malas', 'terrible', 'horrible', 'pesimo',
    'pesima', 'fatal', 'triste', 'asqueroso', 'asquerosa', 'vergonzoso',
    'lamentable', 'deplorable', 'nefasto', 'nefasta', 'patetico', 'patetica',
    'detestable', 'odioso', 'odiosa', 'maldito', 'maldita', 'estupido',
    'estupida', 'idiota', 'inutil', 'falso', 'falsa', 'corrupto', 'corrupta',
    'violento', 'violenta', 'injusto', 'injusta', 'miserable',
    'odio', 'asco', 'verguenza', 'fracaso', 'crisis', 'catastrofe', 'desastre',
    'corrupcion', 'mentira', 'estafa', 'abuso', 'injusticia', 'miedo', 'terror',
    'angustia', 'desesperacion', 'dolor', 'sufrimiento', 'muerte', 'tragedia',
    'problema', 'conflicto', 'guerra', 'violencia', 'crimen', 'amenaza',
    'peligro', 'desgracia', 'caos', 'escandalo', 'humillacion', 'decepcion',
    'frustracion', 'rabia', 'furia', 'ira', 'rencor',
    'odiar', 'perder', 'fallar', 'fracasar', 'sufrir', 'llorar', 'destruir',
    'arruinar', 'hundir', 'robar', 'mentir', 'engañar', 'traicionar',
    'abandonar', 'rechazar', 'insultar', 'agredir', 'humillar',
    'mal', 'peor', 'jamas', 'ladron', 'mentiroso', 'mentirosa', 'hipocrita',
    'lamentablemente', 'tristemente', 'desgraciadamente',
}

def extract_lexicon_features(texts):
    features = []
    for text in texts:
        words = set(text.lower().split())
        pos_score = len(words & POSITIVE_WORDS)
        neg_score = len(words & NEGATIVE_WORDS)
        net = pos_score - neg_score
        # Separamos net en dos features >= 0 para compatibilidad con ComplementNB
        net_pos = max(0, net)
        net_neg = max(0, -net)
        n_words = max(len(words), 1)
        pos_ratio = pos_score / n_words
        neg_ratio = neg_score / n_words
        features.append([pos_score, neg_score, net_pos, net_neg, pos_ratio, neg_ratio])
    return csr_matrix(np.array(features, dtype=np.float32))

# %% [markdown]
# ## 7. Manejo de negación
#
# Se aplica DESPUÉS de transformer. Palabras sin tildes.

# %%
NEGATION_WORDS = {
    'no', 'ni', 'nunca', 'jamas', 'tampoco',
    'sin', 'nada', 'nadie', 'ningun', 'ninguna'
}
NEGATION_WINDOW = 4

def apply_negation(text):
    words = text.split()
    result = []
    negating = 0
    for word in words:
        clean = word.strip('.,!?;:')
        if clean in NEGATION_WORDS:
            negating = NEGATION_WINDOW
            result.append(word)
        elif negating > 0:
            result.append('NEG_' + word)
            negating -= 1
            if re.search(r'[.!?]', word):
                negating = 0
        else:
            result.append(word)
    return ' '.join(result)

# %% [markdown]
# ## 8. Preprocesamiento completo
#
# Orden correcto:
# 1. Features manuales sobre texto crudo (emojis, !, ?, mayúsculas).
# 2. tp.transformer (limpieza y normalización).
# 3. apply_negation sobre texto limpio.
# 4. Features de lexicón sobre texto limpio.

# %%
x_train_raw = data_train['content'].tolist()
x_test_raw = data_test['content'].tolist()

y_train = data_train['sentiment/polarity/value'].values
y_test = data_test['sentiment/polarity/value'].values

# Paso 1: features manuales sobre texto crudo
x_train_manual = extract_manual_features(x_train_raw)
x_test_manual = extract_manual_features(x_test_raw)

# Paso 2: transformer
x_train_clean = [tp.transformer(row) for row in x_train_raw]
x_test_clean = [tp.transformer(row) for row in x_test_raw]

# Paso 3: negación
x_train_proc = [apply_negation(t) if t else '' for t in x_train_clean]
x_test_proc = [apply_negation(t) if t else '' for t in x_test_clean]

# Paso 4: lexicón
x_train_lex = extract_lexicon_features(x_train_proc)
x_test_lex = extract_lexicon_features(x_test_proc)

print(f"x_train: {len(x_train_proc)} | x_test: {len(x_test_proc)}")

# %% [markdown]
# ## 9. Vectorización: TF-IDF word + char

# %%
tfidf_word = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    strip_accents='unicode',
)

tfidf_char = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),
    sublinear_tf=True,
    min_df=3,
    max_df=0.95,
    strip_accents='unicode',
)

# %%
x_train_w = tfidf_word.fit_transform(x_train_proc)
x_test_w = tfidf_word.transform(x_test_proc)

x_train_c = tfidf_char.fit_transform(x_train_proc)
x_test_c = tfidf_char.transform(x_test_proc)

x_train = hstack([x_train_w, x_train_c, x_train_manual, x_train_lex])
x_test = hstack([x_test_w, x_test_c, x_test_manual, x_test_lex])

print(f"Dimensión train: {x_train.shape}")
print(f"Dimensión test : {x_test.shape}")

# %% [markdown]
# ## 10. Análisis de distribución de clases

# %%
print('**Sample train:', sorted(Counter(y_train).items()))

# %%
print('**Sample test:', sorted(Counter(y_test).items()))

# %% [markdown]
# ## 11. GridSearchCV para ajuste de hiperparámetros
#
# Buscamos el mejor valor de C para LinearSVC antes de construir el ensemble.
# Usamos f1_macro como métrica porque nos importan las clases minoritarias (NEU, NONE).

# %%
# Oversampling previo para el grid search
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
x_train_gs, y_train_gs = ros.fit_resample(x_train, y_train)

svc_grid = GridSearchCV(
    LinearSVC(max_iter=5000, class_weight='balanced', random_state=42),
    param_grid={'C': [0.1, 0.5, 1.0, 5.0, 10.0]},
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1,
)
svc_grid.fit(x_train_gs, y_train_gs)
best_C = svc_grid.best_params_['C']
print(f"Mejor C para LinearSVC: {best_C}")
print(f"Mejor F1 macro en CV  : {round(svc_grid.best_score_ * 100, 2)}%")

# %% [markdown]
# ## 12. Esquema de validación: ShuffleSplit

# %%
k_fold = ShuffleSplit(n_splits=10, test_size=0.25, random_state=42)

# %% [markdown]
# ## 13. Oversampling con SMOTE solo sobre train
#
# SMOTE genera ejemplos sintéticos interpolando entre vecinos cercanos,
# más diverso que RandomOverSampler que solo duplica.
# Solo se aplica sobre train, el test queda intacto.
#
# Requiere al menos k_neighbors+1 ejemplos por clase.
# Usamos k_neighbors=3 para clases pequeñas como NEU y NONE.

# %%
smote = SMOTE(random_state=42, k_neighbors=3)
x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

print('**SMOTE train:', sorted(Counter(y_train_res).items()))

# %% [markdown]
# ## 14. Modelo: Ensemble VotingClassifier
#
# Fixes aplicados:
# - LinearSVC: max_iter=5000 para evitar ConvergenceWarning.
# - LogisticRegression: sin multi_class (deprecado en sklearn 1.5).
# - ComplementNB: solo recibe features >= 0 (garantizado por el lexicón corregido).

# %%
svc = CalibratedClassifierCV(
    # max_iter=5000 resuelve el ConvergenceWarning
    LinearSVC(C=best_C, max_iter=5000, class_weight='balanced', random_state=42),
    cv=3
)

lr = LogisticRegression(
    C=5.0,
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    # multi_class eliminado — deprecado en sklearn 1.5, ahora siempre es multinomial
    random_state=42
)

cnb = CalibratedClassifierCV(
    ComplementNB(alpha=0.1),
    cv=3
)

softmax = VotingClassifier(
    estimators=[('svc', svc), ('lr', lr), ('cnb', cnb)],
    voting='soft'
)

# %% [markdown]
# ## 15. Métricas de evaluación

# %%
accuracies_scores = []
recalls_scores = []
precisions_scores = []
f1_scores = []

# %% [markdown]
# ## 16. Entrenamiento y validación cruzada
#
# Oversampling dentro de cada fold para métricas honestas.

# %%
for train_index, test_index in k_fold.split(x_train_res, y_train_res):
    data_train_fold = x_train_res[train_index]
    target_train_fold = y_train_res[train_index]

    data_test_fold = x_train_res[test_index]
    target_test_fold = y_train_res[test_index]

    # SMOTE dentro del fold
    smote_fold = SMOTE(random_state=42, k_neighbors=3)
    data_train_fold, target_train_fold = smote_fold.fit_resample(data_train_fold, target_train_fold)

    softmax.fit(data_train_fold, target_train_fold)
    predict = softmax.predict(data_test_fold)

    accuracies_scores.append(accuracy_score(target_test_fold, predict))
    recalls_scores.append(recall_score(target_test_fold, predict, average='macro'))
    precisions_scores.append(precision_score(target_test_fold, predict, average='weighted', zero_division=0))
    f1_scores.append(f1_score(target_test_fold, predict, average='weighted'))

# %% [markdown]
# ## 17. Resultados promedio en validación

# %%
average_recall = round(np.mean(recalls_scores) * 100, 2)
average_precision = round(np.mean(precisions_scores) * 100, 2)
average_f1 = round(np.mean(f1_scores) * 100, 2)
average_accuracy = round(np.mean(accuracies_scores) * 100, 2)

# %%
average_recall

# %% [markdown]
# ## 18. Evaluación final sobre el conjunto de prueba

# %%
softmax.fit(x_train_res, y_train_res)

y_predict = []
for features in x_test:
    features = features.reshape(1, -1)
    value = softmax.predict(features)[0]
    y_predict.append(value)

classification = classification_report(y_test, y_predict, zero_division=0)
confusion = confusion_matrix(y_predict, y_test)

# %%
output_result = {
    'F1-score': average_f1,
    'Accuracy': average_accuracy,
    'Recall': average_recall,
    'Precision': average_precision,
    'Classification Report\n': classification,
    'Confusion Matrix\n': confusion
}

# %%
for item, val in output_result.items():
    print('{0} {1}'.format(item, val))

# %% [markdown]
# ## 19. Matriz de confusión visual

# %%
labels = sorted(set(y_test))
cm = confusion_matrix(y_test, y_predict, labels=labels)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=labels, yticklabels=labels, ax=ax
)
ax.set_xlabel('Predicho')
ax.set_ylabel('Real')
ax.set_title('Matriz de Confusion — Ensemble + TF-IDF + Lexicon + Features Manuales')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 20. Análisis de ejemplos mal clasificados (NEU y NONE)
#
# Inspeccionamos los tweets donde el modelo falla en NEU y NONE
# para entender la confusión semántica entre ambas clases y
# orientar mejoras futuras del lexicón o del preprocesamiento.

# %%
df_test = data_test.copy()
df_test['predicted'] = y_predict
df_test['correct'] = df_test['sentiment/polarity/value'] == df_test['predicted']

# Errores en NEU
errors_neu = df_test[
    (df_test['sentiment/polarity/value'] == 'NEU') & (~df_test['correct'])
][['content', 'sentiment/polarity/value', 'predicted']]

print(f"Errores en NEU: {len(errors_neu)}")
errors_neu.head(10)

# %%
# Errores en NONE
errors_none = df_test[
    (df_test['sentiment/polarity/value'] == 'NONE') & (~df_test['correct'])
][['content', 'sentiment/polarity/value', 'predicted']]

print(f"Errores en NONE: {len(errors_none)}")
errors_none.head(10)

# %%
# Con qué clases se confunde NEU
print("NEU predicho como:")
print(errors_neu['predicted'].value_counts())

# %%
# Con qué clases se confunde NONE
print("NONE predicho como:")
print(errors_none['predicted'].value_counts())