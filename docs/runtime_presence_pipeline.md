# Runtime Presence Pipeline

## Зачем это нужно

Этот runtime-пайплайн нужен для дешевой детекции появления объекта в зоне приема до запуска основной модели классификации мусора.

Цель:
- не запускать CNN на каждом кадре;
- не добавлять класс `nothing` в текущий classifier;
- получить простой baseline, который можно отладить на ноутбуке и потом перенести на ESP.

Текущий classifier остается 3-class:
- `paper`
- `plastic`
- `other`

## Общая схема

Алгоритм детекции фиксирован и сейчас работает так:

```text
frame
  -> crop ROI
  -> grayscale
  -> resize
  -> GaussianBlur
  -> absdiff with background
  -> threshold
  -> morphology open/close
  -> foreground_ratio
  -> brightness check
  -> hysteresis gate
  -> state machine
```

Идея простая:
- detector следит только за ROI;
- если в ROI стабильно появился объект, pipeline собирает несколько кадров;
- выбирается лучший кадр по максимальному `foreground_ratio`;
- classifier вызывается один раз на объект;
- после этого pipeline ждет, пока сцена снова станет пустой.

## Что относится к runtime, а что только к desktop-debug

### Runtime-ядро

Это часть, которую нужно считать будущей основой для embedded-переноса:

- `configs/runtime/presence_config.py`
- `src/pipeline/presence_detector.py`
- `src/pipeline/frame_selector.py`
- `src/pipeline/runtime_state_machine.py`

### Desktop-only слой

Это нужно для калибровки и локальной отладки на ноутбуке, но не является целевым API для ESP:

- `scripts/calibrate_presence_detector.py`
- `scripts/replay_presence_pipeline.py`
- debug-данные detector-а:
  - `mask`
  - `diff`
  - `roi_small`
- debug-данные selector-а:
  - `meta`
- debug-слой pipeline:
  - `debug`
  - `selected_meta`
  - `selector_scores`
  - `classification`
  - `classifier_input`

### Что не надо тащить на ESP как есть

- загрузку background из `.npy` и картинок;
- replay overlays;
- универсальные file-based desktop helper-утилиты;
- debug payload, нужный только для ноутбука.

Для ESP целевая форма проще:
- маленький заранее откалиброванный background buffer;
- компактная state machine;
- минимальный result/event API.

## Структура файлов

### `configs/runtime/presence_config.py`

Хранит runtime-конфиг:
- ROI;
- размер preprocessing;
- threshold;
- morphology-параметры;
- hysteresis;
- classifier crop/input size;
- cooldown;
- adaptive background flag.

Важно:
- `use_adaptive_background` по умолчанию выключен;
- код adaptive background оставлен в Python-реализации, но baseline для проекта сейчас считается fixed calibrated background.

### `src/pipeline/presence_detector.py`

Содержит несколько маленьких частей с разной ответственностью.

#### `crop_to_roi(...)`

Общий helper для crop по ROI.

#### `FramePreprocessor`

Отвечает только за preprocessing:
- crop;
- grayscale;
- resize;
- blur.

#### `BackgroundModel`

Отвечает только за фон:
- сборка по пустым кадрам;
- хранение reference;
- `absdiff`;
- опциональное медленное обновление.

#### `PresenceGate`

Отвечает только за hysteresis:
- `enter_frames`
- `exit_frames`

#### `PresenceDetector`

Собирает всё вместе:
- подготавливает кадр;
- считает diff;
- строит mask;
- считает `foreground_ratio`;
- проверяет brightness;
- обновляет hysteresis gate;
- при необходимости обновляет background только в empty-сцене.

### `src/pipeline/frame_selector.py`

`BestFrameSelector` хранит несколько последних кандидатов и выбирает лучший по score.

Сейчас score:
- `foreground_ratio`

Это минимальная версия baseline.

`meta` хранится только для desktop replay/debug, не как embedded interface.

### `src/pipeline/runtime_state_machine.py`

Высокоуровневый orchestration-слой.

Состояния:
- `EMPTY`
- `OBJECT_ENTERING`
- `OBJECT_PRESENT`
- `WAIT_UNTIL_EMPTY`

Логика:
- в `EMPTY` detector ждет появления сигнала;
- в `OBJECT_ENTERING` ждем подтверждение;
- в `OBJECT_PRESENT` накапливаем кадры;
- после накопления выбираем лучший кадр и один раз вызываем classifier;
- переходим в `WAIT_UNTIL_EMPTY`;
- возвращаемся в `EMPTY`, когда сцена действительно очистилась.

## Как откалибровать background

### 1. Настроить ROI

Сначала нужно выставить реальный ROI в конфиге:

`configs/runtime/presence_config.py`

Без корректного ROI калибровка порогов бессмысленна.

### 2. Собрать пустые кадры

Нужна папка с пустой сценой, например:

```text
data/empty_frames/
```

Желательно:
- 30-100 кадров;
- та же камера;
- то же положение камеры;
- то же освещение;
- в зоне ROI нет объекта.

### 3. Запустить калибровку

```bash
uv run -m scripts.calibrate_presence_detector ^
  --input_dir data/empty_frames ^
  --output_path experiments/presence/background_reference.npy ^
  --debug_preview_path experiments/presence/background_preview.png
```

Результат:
- `background_reference.npy` — reference background для detector-а;
- `background_preview.png` — визуальный preview;
- в консоли — статистика по brightness, diff и foreground ratio.

### 4. Что смотреть после калибровки

Если на пустых кадрах уже большой `foreground_ratio`, то проблема обычно в одном из пунктов:
- ROI выбран неудачно;
- слишком низкий `diff_threshold`;
- слишком низкий `min_foreground_ratio`;
- нестабильный свет;
- сильные блики.

## Как прогонять replay

Для папки кадров:

```bash
uv run -m scripts.replay_presence_pipeline ^
  --input_dir data/debug_sequences/sequence_01 ^
  --background_path experiments/presence/background_reference.npy ^
  --save_dir experiments/presence/replay_sequence_01
```

Для видео:

```bash
uv run -m scripts.replay_presence_pipeline ^
  --video_path data/debug_sequences/sequence_01.mp4 ^
  --background_path experiments/presence/background_reference.npy ^
  --save_dir experiments/presence/replay_sequence_01
```

Что делает replay:
- прогоняет последовательность через `RuntimePipeline`;
- печатает смену состояний;
- показывает момент вызова classifier;
- сохраняет debug overlays;
- сохраняет classifier crop.

## Как с этим работать команде

Практический workflow такой:

1. Зафиксировать камеру и ROI.
2. Откалибровать `background_reference.npy`.
3. Прогнать несколько empty-sequence сценариев.
4. Прогнать несколько object-present сценариев.
5. Подкрутить thresholds в `presence_config.py`.
6. Только после этого подключать реальный classifier adapter.

Рекомендуется:
- не смешивать detector и classifier в один класс;
- не использовать desktop debug-поля как часть будущего embedded contract;
- не включать adaptive background по умолчанию до момента, когда команда осознанно решит, что он нужен.

## Базовые параметры, которые обычно крутят

В первую очередь:
- `roi`
- `diff_threshold`
- `min_foreground_ratio`

Потом:
- `enter_frames`
- `exit_frames`
- `cooldown_frames`

Остальное уже вторично.

## Что будет переноситься на ESP

Целевая embedded-версия должна взять только runtime-ядро:

- preprocessing по ROI;
- background buffer маленького размера;
- threshold + morphology;
- `foreground_ratio`;
- brightness check;
- hysteresis gate;
- state machine;
- выбор лучшего кадра по score.

### В каком виде лучше переносить

Не как прямую копию desktop Python API, а как компактные embedded сущности:

- `Roi`
- `PresenceConfig`
- `BackgroundModel`
- `PresenceGate`
- `PresenceDetector`
- `PipelineState`
- `RuntimePipeline`

### Что использовать на ESP вместо desktop background loading

Не файл `.npy`, а уже готовый маленький background buffer, откалиброванный заранее на ноутбуке.

То есть pipeline переносится как:
- fixed config;
- fixed ROI;
- precomputed small background image;
- small integer buffers;
- state machine.

## Что важно помнить

- текущий detector специально rule-based;
- desktop debug-слой нужен для калибровки и replay;
- embedded API должен быть уже, проще и без file/debug-логики;
- classifier должен вызываться только по событию, а не на каждом кадре.
