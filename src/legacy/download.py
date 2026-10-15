"""Скачивание фиксированной версии открытого датасета с проверкой SHA-256."""
import requests
from prepare import OUT, REVISION, EXPECTED_SHA256, digest


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT/'listens.parquet'
    if target.exists() and digest(target) == EXPECTED_SHA256:
        print('Датасет уже скачан, SHA-256 совпадает.')
        return
    url = f'https://huggingface.co/datasets/yandex/yambda/resolve/{REVISION}/flat/50m/listens.parquet'
    tmp = target.with_suffix('.download')
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in response.iter_content(8*1024*1024):
                f.write(chunk)
    assert digest(tmp) == EXPECTED_SHA256, 'SHA-256 скачанного файла не совпадает'
    tmp.replace(target)
    print('Yambda-50M скачан, SHA-256 подтверждён.')


if __name__ == '__main__':
    main()
