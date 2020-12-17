from distutils.core import setup
setup(
  name = 'effunet',         
  packages = ['effunet'],   
  version = '0.1',      
  license='MIT',
  description = 'UNet segmentation model with an efficiennet encoder',   
  author = 'Pranshu Mishra',                   
  author_email = 'pranshumshr.04@gmail.com',      
  url = 'https://github.com/pranshu97/effunet',   
  download_url = '',
  keywords = ['UNet', 'EfficienNet', 'Segmentation'],
  install_requires=[            
          'torch',
          'torchvision',
          'efficientnet_pytorch'
      ],
  classifiers=[
    'Development Status :: Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',  
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)