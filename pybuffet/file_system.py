import os
import yaml, sys

def read_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print("Text file not found:", file_path)
        raise

def read_yaml_file(file_path):
    try:
        with open(file_path, encoding='utf-8') as f:
            data = yaml.safe_load(f)  # parses YAML into dict/list/str/etc.
            return data
    except FileNotFoundError:
        print("YAML file not found:", file_path)
        raise
    except yaml.YAMLError as e:
        print("Invalid YAML:", e)
        raise

__all__ = ['read_txt_file', 'read_yaml_file']