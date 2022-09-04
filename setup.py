from setuptools import setup

setup(
    name='video_speaker_analysis',
    version='1.0',
    packages=['src', 'src.database', 'src.owl_data', 'src.zoom_data'],
    install_requires=['numpy', 'dlib', 'face_recognition', 'pytesseract', 'opencv-python', 'fuzzywuzzy', 'tqdm',
                      'scikit-learn'],
    scripts=['bin/zoom_video'],
    url='',
    license='MIT',
    author='Ishaan Narain',
    author_email='ishaannarain2022@u.northwestern.edu',
    description='A module for extracting speaker data from Zoom and Owl Videos'
)
