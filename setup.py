from distutils.core import setup
setup(
  name = 'effunet',         
  packages = ['effunet'],   
  version = '1.1.0',      
  license='MIT',
  description = 'UNet segmentation model with an efficientnet encoder',   
  author = 'Pranshu Mishra',                   
  author_email = 'pranshumshr.04@gmail.com',      
  url = 'https://github.com/pranshu97/effunet',   
  download_url = 'https://github.com/pranshu97/effunet/archive/v1.1.0.tar.gz',
  keywords = ['UNet', 'EfficientNet', 'Segmentation'],
  install_requires=[            
          'torch>=1.7.0',
          'torchvision>=0.8.1',
          'efficientnet_pytorch>=0.7.0'
      ],
  classifiers=[
    'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',  
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)