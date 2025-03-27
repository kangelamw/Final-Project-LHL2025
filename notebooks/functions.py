# Imports
import os
import re
import pandas as pd


# Etc.
def generate_file_structure(start_path, indent=''):
  """
  Generate a file structure tree for a given directory path... to copy&paste on the README.md file...
  """
  file_structure = ''
  for item in os.listdir(start_path):
      if item == '.git':
        continue # Skip the .git folder
      
      item_path = os.path.join(start_path, item)
      if os.path.isdir(item_path):
        file_structure += f'{indent}├── {item}\n'
        file_structure += generate_file_structure(item_path, indent + '│   ')
      else:
        file_structure += f'{indent}├── {item}\n'
  return file_structure

# Cleaning
"""
def remove_stopwords(text):
  import nltk
  try:
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    
    return [token for token in text if token not in stop_words]
  except LookupError:
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    
    return [token for token in text if token not in stop_words]
""" # Didn't use
