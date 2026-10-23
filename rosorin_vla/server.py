"""Optional local inference service; tunnel over SSH for a separate GPU host."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--port',type=int,default=8767)
    args=parser.parse_args()
    from .policy import LocalPolicy
    import numpy as np
    policy=LocalPolicy(args.model_dir,args.device)
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def reply(self,code,payload):
            body=json.dumps(payload,allow_nan=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            self.reply(200,{'state':'ready'}) if self.path=='/health' else self.reply(404,{'error':'NOT_FOUND'})
        def do_POST(self):
            if self.path!='/predict':return self.reply(404,{'error':'NOT_FOUND'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=2_000_000:raise ValueError('INVALID_BODY_SIZE')
                request=json.loads(self.rfile.read(length))
                if set(request)!={'request_id','initial_npz_b64'}:raise ValueError('INITIAL_RGB_AND_STATE_ONLY')
                if not isinstance(request['request_id'],str) or len(request['request_id'])>100:raise ValueError('INVALID_REQUEST_ID')
                with np.load(io.BytesIO(base64.b64decode(request['initial_npz_b64'],validate=True)),allow_pickle=False) as data:
                    if set(data.files)!={'rgb','state'}:raise ValueError('INITIAL_RGB_AND_STATE_ONLY')
                    obs={'rgb':data['rgb'],'state':data['state']}
                if not lock.acquire(timeout=1):return self.reply(409,{'error':'INFERENCE_BUSY'})
                try:result=policy.predict(obs,request['request_id'])
                finally:lock.release()
                self.reply(200,result)
            except Exception as exc:self.reply(400,{'error':str(exc)})
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Model ready at http://127.0.0.1:{args.port}; loopback only.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=='__main__':main()
