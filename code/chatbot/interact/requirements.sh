curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

git clone https://github.com/alphacep/vosk-api.git
cd vosk-api/python
python setup.py install

sudo apt-get install libportaudio2
