from setuptools import setup, find_packages


setup(name='WAKD',
      version='1.0',
      description='Open-source toolbox for Image-based Localization',
      author_email='yifanxu98@163.com',
      url='https://github.com/XuYifan98/WAKD',
      license='MIT',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
          'Image Localization',
          'Image Matching'
      ])
