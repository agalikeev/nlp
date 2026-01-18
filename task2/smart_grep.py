import argparse
import gzip
import os
import re
import shutil
from pathlib import Path

from difflib import get_close_matches

import pymorphy3
import requests
from gensim.models import KeyedVectors

CACHE_DIR = os.path.expanduser("cache/")
MODEL_FILE = "w2v_vectors_news.txt"
MODEL_URL = "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/FaDZnO1IjNvjvw"


def download_model(url: str = MODEL_URL) -> str | None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    gz_path = os.path.join(CACHE_DIR, MODEL_FILE + ".gz")
    path = os.path.join(CACHE_DIR, MODEL_FILE)

    try:
        print(f"Загружаем модель из {url}...")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(gz_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        if not os.path.isfile(gz_path) or os.path.getsize(gz_path) == 0:
            print("Ошибка. Файл не был загружен корректно.")
            return None

        print("Распаковываем архив")
        try:
            with gzip.open(gz_path, "rb") as gz, open(path, "wb") as out:
                shutil.copyfileobj(gz, out)
        except (OSError, gzip.BadGzipFile) as gz_err:
            print(f"Ошибка при распаковке GZIP: {gz_err}")
            return None
        finally:
            if os.path.exists(gz_path):
                os.remove(gz_path)

        print(f"Модель успешно загружена: {path}")
        return path

    except requests.exceptions.RequestException as net_err:
        print(f"Ошибка сети при загрузке модели: {net_err}")
    except IOError as io_err:
        print(f"Ошибка при записи файла: {io_err}")
    except Exception as e:
        print(f"Непредвиденная ошибка при загрузке модели: {e}")

    return None


def load_model() -> KeyedVectors | None:
    path = os.path.join(CACHE_DIR, MODEL_FILE)
    try:
        if os.path.isfile(path):
            print(f"Загружаем модель из {path}")
            model = KeyedVectors.load_word2vec_format(
                path, encoding="utf-8", unicode_errors="ignore"
            )
            print("Модель успешно загружена")
            return model
        else:
            print("Локальная модель не найдена, пробуем скачать заново")
            dpath = download_model()
            if dpath and os.path.isfile(dpath):
                model = KeyedVectors.load_word2vec_format(
                    dpath, encoding="utf-8", unicode_errors="ignore"
                )
                print("Модель успешно загружена после скачивания.")
                return model
            else:
                print("Не удалось получить путь к скачанной модели.")
    except (ValueError, OSError) as read_err:
        print(f"Ошибка при чтении или распознавании модели: {read_err}")
    except Exception as e:
        print(f"Непредвиденная ошибка при загрузке модели: {e}")

    print("Модель не загружена.")
    return None


def normalize(word, morph):
    return morph.parse(word)[0].normal_form


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^а-яёa-z\s-]", " ", text)
    return re.findall(r"\b[а-яёa-z][а-яёa-z-]*\b", text)


def mygrep(pattern: str, file_path: str | Path,
           w2v_model: KeyedVectors, topn=8, threshold=0.6, sim_threshold=0.7):
    morph = pymorphy3.MorphAnalyzer()
    pattern_norm = normalize(pattern, morph)

    candidates: list[str] = []

    if pattern_norm in w2v_model:
        similar = w2v_model.most_similar(pattern_norm, topn=topn)
        candidates = [pattern_norm] + [w for w, score in similar if score >= threshold]

    else:
        print(f"'{pattern_norm}' отсутствует в модели, ищем по частям ...")
        parts = re.split(r"[-_]", pattern_norm)
        found_any = False
        for part in parts:
            if part == "":
                continue
            if part in w2v_model:
                found_any = True
                similar = w2v_model.most_similar(part, topn=topn)
                sub_candidates = [part] + [w for w, score in similar if score >= threshold]
                print(f"Для части '{part}' найдены кандидаты: {', '.join(sub_candidates)}")
                candidates.extend(sub_candidates)
        if not found_any:
            print(f"Ни одной части из '{pattern_norm}' нет в модели.")
            candidates = get_close_matches(pattern_norm, w2v_model.key_to_index.keys(), n=3, cutoff=0.6)

    candidates = sorted(set(candidates))
    print(f"Базовые слова: {', '.join(candidates)}")

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            words = tokenize(line)
            lemmas = [normalize(w, morph) for w in words]

            for w, l in zip(words, lemmas):
                if any(l.startswith(c) or l == c for c in candidates):
                    print(f"найдено слово: {w}  (лемма: {l})")
                    print(f"line {i}: {line.strip()}")
                    break
            else:
                match_found = False
                for w, l in zip(words, lemmas):
                    if l in w2v_model:
                        for c in candidates:
                            try:
                                sim = w2v_model.similarity(l, c)
                                if sim > sim_threshold:
                                    print(f"Похожее слово: {w}  (лемма: {l}, sim={sim:.3f})")
                                    print(f"line {i}: {line.strip()}")
                                    match_found = True
                                    break
                            except KeyError:
                                continue
                    if match_found:
                        break


def main():
    parser = argparse.ArgumentParser(description="My smart grep")
    parser.add_argument("file_path", type=Path)
    parser.add_argument("query", type=str)
    args = parser.parse_args()

    model = load_model()
    if model is None:
        print("Модель не загружена")
        return

    mygrep(args.query, args.file_path, model)


if __name__ == "__main__":
    main()
