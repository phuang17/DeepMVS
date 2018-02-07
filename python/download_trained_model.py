import os
import sys
import subprocess

def download_trained_model(path = None):
	if path is None:
		path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")
	if not os.path.isdir(path):
		os.mkdir(path)
	print "Downloading trained model..."
	subprocess.call(
		"cd {:} ;".format(path) + 
		"wget -O DeepMVS_final.model https://www.dropbox.com/s/dcaip7n3z0yk7lx/DeepMVS_final.model?dl=1 ;",
		shell = True
	)
	print "Successfully downloaded trained model."

if __name__ == "__main__":
	download_trained_model()

