# parzen_window

[Русский](#русский) | [English](#english)

---

## English

`parzen_window` is a Python library implementing the Parzen window method for
classification, built on top of `numpy` and `scikit-learn` (`BaseEstimator`,
`ClassifierMixin`).

### Features

- Five kernel shapes: `gaussian`, `epanechnikov`, `quartic`, `triangular`, `rectangular`.
- **Adaptive bandwidth** — each training point can get its own kernel width,
  derived from the distance to its `k`-th nearest neighbor, instead of a
  single fixed `h` for the whole dataset (a variable-kernel / "balloon"
  estimator). This is the library's scientific contribution over plain
  fixed-bandwidth Parzen window implementations: it widens the kernel in
  sparse regions and narrows it in dense ones automatically.
- **Incremental learning** — `partial_fit()` extends the model with a new
  batch of data without discarding what it already learned.
- **Dataset compaction** — `compact()` reduces memory usage by removing
  training points that are deep inside their own class's territory (found
  via convex-neighborhood analysis: a point is dropped only if all its
  nearest neighbors share its label *and* it is far from the nearest point
  of a different class). This shrinks the stored dataset with minimal
  impact on the decision boundary.
- **Thread-safe compaction with a call scheduler** — while `compact()` is
  running, the training data is locked for writes. `predict()` and
  `partial_fit()` calls from other threads do not fail during this time:
  they are queued and executed automatically as soon as compaction
  finishes.
- **`.npz` persistence** — `save()`/`load()` for backups and moving a model
  between machines.
- Fully compatible with `pickle`/`joblib`.

### Installation

```bash
pip install parzen_window
```

### Quick start

```python
from parzen_window import ParzenWindowClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = ParzenWindowClassifier(h=1.0, kernel="quartic")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

Incremental learning:

```python
model = ParzenWindowClassifier(h=0.5, kernel="gaussian")
model.fit(X_batch_1, y_batch_1)
model.partial_fit(X_batch_2, y_batch_2)  # extends, does not discard X_batch_1
```

Adaptive bandwidth:

```python
model = ParzenWindowClassifier(
    h=1.0, kernel="gaussian", adaptive_bandwidth=True, bandwidth_neighbors=5
)
model.fit(X_train, y_train)
```

Compacting a large dataset to cut memory usage by ~25%:

```python
removed = model.compact(0.25)  # returns the number of points actually removed
```

Backups:

```python
model.save("model.npz")
restored = ParzenWindowClassifier.load("model.npz")
```

### Development

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --dev        # install dependencies
uv run ruff check .  # lint
uv run ruff format .  # format
uv build              # build sdist + wheel
```

---

## Русский

`parzen_window` — библиотека на Python, реализующая метод Парзеновского окна
для классификации, построенная поверх `numpy` и `scikit-learn`
(`BaseEstimator`, `ClassifierMixin`).

### Возможности

- Пять видов ядер: `gaussian`, `epanechnikov`, `quartic`, `triangular`, `rectangular`.
- **Адаптивная ширина окна** — каждая точка обучающей выборки может получить
  собственную ширину ядра, вычисленную по расстоянию до её `k`-го ближайшего
  соседа, вместо единого фиксированного `h` на весь датасет (вариант
  variable-kernel / "balloon"-оценки). Это научная новизна библиотеки по
  сравнению с обычными реализациями Парзеновского окна с фиксированной
  шириной: окно автоматически расширяется в разреженных областях и
  сужается в плотных.
- **Дообучение** — `partial_fit()` расширяет модель новым батчем данных, не
  теряя уже накопленное.
- **Разрядка датасета** — `compact()` уменьшает потребление памяти, удаляя
  точки обучающей выборки, которые лежат глубоко на территории своего
  класса (метод анализа выпуклых окрестностей: точка удаляется, только
  если все её ближайшие соседи имеют ту же метку класса *и* она далека от
  ближайшей точки другого класса). Это сокращает хранимый датасет с
  минимальным влиянием на границу между классами.
- **Потокобезопасная разрядка с планировщиком вызовов** — пока выполняется
  `compact()`, обучающие данные заблокированы для записи. Вызовы
  `predict()` и `partial_fit()` из других потоков в это время не падают:
  они встают в очередь и выполняются автоматически сразу после завершения
  разрядки.
- **Сохранение в `.npz`** — `save()`/`load()` для бэкапов и переноса модели
  между устройствами.
- Полная совместимость с `pickle`/`joblib`.

### Установка

```bash
pip install parzen_window
```

### Быстрый старт

```python
from parzen_window import ParzenWindowClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = ParzenWindowClassifier(h=1.0, kernel="quartic")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

Дообучение:

```python
model = ParzenWindowClassifier(h=0.5, kernel="gaussian")
model.fit(X_batch_1, y_batch_1)
model.partial_fit(X_batch_2, y_batch_2)  # расширяет, не теряя X_batch_1
```

Адаптивная ширина окна:

```python
model = ParzenWindowClassifier(
    h=1.0, kernel="gaussian", adaptive_bandwidth=True, bandwidth_neighbors=5
)
model.fit(X_train, y_train)
```

Разрядка большого датасета, чтобы сократить потребление памяти примерно на 25%:

```python
removed = model.compact(0.25)  # возвращает число реально удалённых точек
```

Бэкапы:

```python
model.save("model.npz")
restored = ParzenWindowClassifier.load("model.npz")
```

### Разработка

Проект использует [uv](https://docs.astral.sh/uv/) для управления зависимостями.

```bash
uv sync --dev         # установить зависимости
uv run ruff check .   # линтер
uv run ruff format .  # форматирование
uv build               # сборка sdist + wheel
```
