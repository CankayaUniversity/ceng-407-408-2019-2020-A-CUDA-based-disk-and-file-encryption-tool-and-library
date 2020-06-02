# A CUDA Based AES-256-CTR File Encryption


A CUDA Based AES-256-CTR File Encryption is a project. We use PC's CPU and GPU for speed. It encrypts and decrypts in CPU or GPU according to file size. And your files will be securely encrypted and decrypted with the AES algorithm and CTR mode. With our project users can encrypt/decrypt their files faster from other encryption/decryption libraries/projects.

## Getting Started
For install our project:

1.  With this link go to our project main page. 
 - https://github.com/CankayaUniversity/ceng-407-408-2019-2020-A-CUDA-based-disk-and-file-encryption-tool-and-library.git
 
2. Click Clone Or Download option. Then select Download Zip option and extract the zip file.
3. The project can only run on linux operating system.
4. You need to install Nvidia Cuda on your computer for running the project.  You can install CUDA on your computer by running the following codes in the terminal.
 - wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
 - sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
 - sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
 - sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
 - sudo apt-get update
 - sudo apt-get -y install cuda
5. Then you need to create a text file for input,a text file for key and an text file for output to run the project.
6. You can open the folder and can run .c and .cu files on the terminal or Nvidia NSight Eclipse Program for encryption your files.
