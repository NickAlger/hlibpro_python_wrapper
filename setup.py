from setuptools import setup

setup(name = 'hlibpro_python_wrapper',
      version = '0.1dev',
      description = 'Python wrapper for HLIBPro',
      author = 'Nick Alger (HLIBPro by Dr. Ronald Kriemann)',
      author_email = 'nalger225@gmail.com',
      url = 'https://github.com/NickAlger/hlibpro_python_wrapper',
      long_description = '''
       HLIBPro is a C++ hierarchical matrix package written by Dr. Ronald Kriemann. 
       This library provides a (very incomplete) set of bindings and helper functions 
       for using HLIBPro from within python. 
       ''',
      packages=['hlibpro_python_wrapper'],
      # package_dir={'hlibpro_wrapper': '.'},
      include_package_data=True,
      # distclass=BinaryDistribution,
      package_data={'hlibpro_python_wrapper': ['hlibpro_bindings.so']},
      )
