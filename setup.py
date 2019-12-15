
from setuptools import setup, find_packages

metadata = {
    'name': 'easytrain',
    'packages': find_packages(),
    'version': '0.2.3',
    'description': 'machine learning utility for lazy man',
    'python_requires': '>=3.5',
    'author': 'Shuhei Hirata',
    'author_email': 'sh7916@gmail.com',
    'license': 'MIT',
    'url': 'https://github.com/hrtshu/easytrain.git',
    'install_requires': [
        'numpy',
        'scikit-learn>=0.21.0',
        'keras>=2.0.0',
    ],
}

setup(**metadata)
