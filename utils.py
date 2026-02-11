import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ssl

def create_ssl_compatible_session():
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    class SSLAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            try:
                ssl_context.minimum_version = ssl.TLSVersion.TLSv1
                ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3
            except AttributeError:
                pass
            try:
                ssl_context.set_ciphers('DEFAULT:@SECLEVEL=1')
            except ssl.SSLError:
                try:
                    ssl_context.set_ciphers('ALL:!aNULL:!eNULL:!EXPORT:!DES:!RC4:!MD5:!PSK:!SRP:!CAMELLIA')
                except ssl.SSLError:
                    pass
            kwargs['ssl_context'] = ssl_context
            return super().init_poolmanager(*args, **kwargs)
    adapter = SSLAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.verify = False
    return session
