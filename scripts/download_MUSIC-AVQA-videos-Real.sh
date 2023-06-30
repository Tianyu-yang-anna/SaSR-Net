#!/bin/bash

# Download zip dataset from Google Drive
filename='MUSIC-AVQA-videos-Real.zip'
fileid='1Ovj5Ay8rDXaPy57CNCHes0A99S43lPBy'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

