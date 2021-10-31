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

