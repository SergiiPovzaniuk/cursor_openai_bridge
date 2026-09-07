import uvicorn

from host.config import CONFIG

if __name__ == "__main__":
    kwargs = dict(host=CONFIG.host, port=CONFIG.port, log_level=CONFIG.log_level.lower())
    if CONFIG.tls_cert_path and CONFIG.tls_key_path:
        kwargs["ssl_certfile"] = CONFIG.tls_cert_path
        kwargs["ssl_keyfile"] = CONFIG.tls_key_path
    uvicorn.run("host.app:app", **kwargs)
