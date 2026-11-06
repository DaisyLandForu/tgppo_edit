"""Verified HTTPS openers for artifact hosts; preserve TLS verification on fallback."""
import ssl
import urllib.request


def make_openers():
    context = ssl.create_default_context(cafile="/etc/ssl/certs/ca-certificates.crt")
    return [
        urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context)),
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=context)),
    ]


def open_url(request, timeout=30, prefer_proxy=False):
    readers = make_openers()
    if prefer_proxy:
        readers.reverse()
    for index in range(4):
        reader = readers[index % len(readers)]
        try:
            return reader.open(request, timeout=timeout)
        except Exception:
            if index == 3:
                raise
