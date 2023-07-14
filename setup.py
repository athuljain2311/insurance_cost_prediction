from setuptools import find_packages,setup

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path):
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [requirement.replace('\n','') for requirement in requirements]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='insurance_cost_prediction',
    version='0.0.1',
    author='Athul Jain',
    author_email='athuljain2311@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)