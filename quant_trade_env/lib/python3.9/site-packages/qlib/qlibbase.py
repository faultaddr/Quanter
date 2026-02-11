class QLibBase():

    version=''

    def  __init__(self,ver='0.0.1'):
        self.version = ver

    def doSay(self,message="Called from QLibBase"):
        return f"{message} @version={self.version}"

import urllib.request
import requests
import urllib.parse
import json
#Some custom exceptions
class Error(Exception):
    pass
class ValidationError(Error):
    pass

#Adapter class acts as interface @qadapter.py
#Facade class acts as common entry point for all useful qclasses ?



class QApi():
    def __init__(self,url):
        self.url=url


    def qhttp_request(self,query=None,method=None, headers={}, data=None):
    #Perform an HTTP request and return the associated response
        url= self.url
        parts = (urllib.parse.urlparse(url))
        print(f'got parts={parts}, type={type(parts)}')
        print(f'extract out got scheme={parts.scheme}, url={parts.netloc}')

        url = parts.geturl()
        print(f'got url={url}')

        r = urllib.request.Request(url=url, method=method,
                               headers=headers,
                               data=data)

        with urllib.request.urlopen(r) as resp:
            msg, resp = resp.info(), resp.read()


        return msg, resp
    #login to use api servicing
    #return token
    def apilogin(self,username,password,encoder='utf-8'):
        url_login = self.url
        data = {'username':username,'password':password}
        req = requests.post(url_login, headers=None,data=json.dumps(data).encode(encoder))
        try:

            if req.status_code not in [400,401,500]:
                #print(req.text)
                a_jwt= json.loads(req.text)
                access_token= a_jwt['access_token']
                return access_token
            else:
                raise ValidationError
        except ValidationError:
            print(f'{json.loads(req.content)}')

    def make_headers(self,token):

        bearer = 'Bearer'+' '+token
        headers={"Authorization":bearer}
        return headers
    @staticmethod
    def get_resource(resourceUrl,**kwargs):
        req= requests.get(resourceUrl,headers=kwargs)
        print(f'get res called {req}')
        try:
            return req.text
        except Exception as ex:
            print(ex)

