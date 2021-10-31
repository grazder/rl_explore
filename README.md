# Prod.Stories RL (HW 5)

## Подготовка 

1. Установка всех пакетов:
`pip install -r requirements.txt`

2. Клонирование подмодулей
`git submodule update --remote`
   
## Обучение

`python train.py --config train_config.yaml`
   
`train_config.yaml` - файл со всеми параметрами в формате HyperPyYaml

## Валидация

`python eval.py --config train_config.yaml --checkpoint <CHECKPOINT_PATH> --n-runs <N_RUNS> --output <OUTPUT_FOLDER>`

`checkpoint` - путь до нужного чекпоинта

`n-runs` - число экспериментов

`output` - путь, куда будут сохраняться все результаты экспериментов

## Описание эксперимента

Был использован PPO из лекции. Это хороший подход, который можно использовать в качестве бейзлайна.
В качестве функции вознаграждения было выбрана следующая функция:

`reward += - 1 / self._max_steps + is_new * self._reward_a + (is_new - 1) * self._reward_b`

Помимо обычной награды здесь добавлен штраф за время, 
добавлено вознаграждение за то, что агент пришел в новую клетку, 
и штрав за то что, агент проходит по клетке, в которой уже был.

Ссылка на эксперименты в wandb: https://wandb.ai/grazder/prod-rl-hw

## Примеры работы агента

Чекпоинт - 197 итерация

![Run 1](./results/gifs/run-1.gif)
![Run 2](./results/gifs/run-2.gif)
![Run 3](./results/gifs/run-3.gif)
![Run 4](./results/gifs/run-4.gif)
![Run 5](./results/gifs/run-5.gif)