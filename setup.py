from setuptools import setup, find_packages

setup(
    name='src',   # hoặc 'eyes_tracking' nếu muốn tên gọn
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
