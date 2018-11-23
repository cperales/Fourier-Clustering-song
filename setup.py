from setuptools import setup

setup(
    name="foucluster",
    description="FouCluster",
    version="1.1",
    author="Carlos Perales",
    author_email="sir.perales@gmail.com",
    packages=['foucluster'
              ],
    zip_safe=False,
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      'sklearn',
                      'seaborn'
                      ],
    include_package_data=True,
    setup_requires=[],
    tests_require=[],
    extras_require={
        'docs': [
            'sphinx'
        ]
    },
)

