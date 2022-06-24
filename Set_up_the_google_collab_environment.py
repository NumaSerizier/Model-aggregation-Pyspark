# we connect google collab to Drive

from google.colab import drive
drive.mount('/content/drive')

# we open our file

tar -xzvf "/content/drive/My Drive/my_file.csv.gz" "/content/drive/My Drive/pyspatktp"

!gunzip "/content/drive/Mon Drive/my_file.csv"
