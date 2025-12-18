# cbf-server
A collection of python utilities to manage CBF images.

### inference server
```
# install dependencies
conda install -c conda-forge pytorch torchvision fastapi uvicorn numpy pillow certifi

# test your certificates
python
>>> import certifi
... print(certifi.where())

# grab the pem file and setup environment
export SSL_CERT_FILE=/path/miniconda3/envs/images/lib/python3.13/site-packages/certifi/cacert.pem
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

# Then, comeback to python and import ML models
>>> from torchvision.models import resnet18, ResNet18_Weights
... resnet18(weights=ResNet18_Weights.DEFAULT)

# start inference server
uvicorn embedding_service:app --host 0.0.0.0 --port 8888
```
