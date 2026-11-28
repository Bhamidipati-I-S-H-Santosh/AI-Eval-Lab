"""Object storage: Amazon S3 (and S3-compatible: MinIO, Ceph, R2), Azure Blob
Storage, Google Cloud Storage. Each uses the vendor's standard credential
chain (IAM role / managed identity / workload identity); no keys in config.

Every backend accepts an already-built `client` (useful for tests and for
custom endpoints)."""
from __future__ import annotations

from . import StorageBackend


class _Prefixed(StorageBackend):
    def __init__(self, prefix: str = ""):
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""

    def _k(self, key: str) -> str:
        return self.prefix + key

    def _strip(self, name: str) -> str:
        return name[len(self.prefix):] if self.prefix and name.startswith(self.prefix) else name


class S3Storage(_Prefixed):
    def __init__(self, bucket: str, prefix: str = "evalkit/", region: str | None = None,
                 endpoint_url: str | None = None, kms_key_id: str | None = None, client=None, **_):
        super().__init__(prefix)
        if client is None:
            import boto3
            client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url)
        self.client, self.bucket, self.kms_key_id = client, bucket, kms_key_id

    def write_text(self, key, text):
        extra = {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": self.kms_key_id} if self.kms_key_id \
            else {"ServerSideEncryption": "AES256"}
        self.client.put_object(Bucket=self.bucket, Key=self._k(key), Body=text.encode("utf-8"),
                               ContentType="text/plain; charset=utf-8", **extra)

    def read_text(self, key):
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key=self._k(key))
        except Exception as e:
            if _is_missing(e):
                return None
            raise
        return obj["Body"].read().decode("utf-8")

    def list(self, prefix):
        out, token = [], None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": self._k(prefix)}
            if token:
                kw["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kw)
            out += [self._strip(o["Key"]) for o in resp.get("Contents", [])]
            if not resp.get("IsTruncated"):
                return sorted(out)
            token = resp.get("NextContinuationToken")

    def delete(self, key):
        self.client.delete_object(Bucket=self.bucket, Key=self._k(key))


class AzureBlobStorage(_Prefixed):
    def __init__(self, container: str, account_url: str | None = None, prefix: str = "evalkit/",
                 connection_string: str | None = None, client=None, **_):
        super().__init__(prefix)
        if client is None:
            from azure.storage.blob import BlobServiceClient
            if connection_string:
                svc = BlobServiceClient.from_connection_string(connection_string)
            else:
                from azure.identity import DefaultAzureCredential
                svc = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
            client = svc.get_container_client(container)
        self.client = client

    def write_text(self, key, text):
        self.client.upload_blob(self._k(key), text.encode("utf-8"), overwrite=True)

    def read_text(self, key):
        try:
            return self.client.download_blob(self._k(key)).readall().decode("utf-8")
        except Exception as e:
            if _is_missing(e):
                return None
            raise

    def list(self, prefix):
        return sorted(self._strip(b.name) for b in self.client.list_blobs(name_starts_with=self._k(prefix)))

    def delete(self, key):
        try:
            self.client.delete_blob(self._k(key))
        except Exception as e:
            if not _is_missing(e):
                raise


class GCSStorage(_Prefixed):
    def __init__(self, bucket: str, prefix: str = "evalkit/", project: str | None = None, client=None, **_):
        super().__init__(prefix)
        if client is None:
            from google.cloud import storage
            client = storage.Client(project=project)
        self.client = client
        self.bucket = client.bucket(bucket)

    def write_text(self, key, text):
        self.bucket.blob(self._k(key)).upload_from_string(text.encode("utf-8"),
                                                          content_type="text/plain; charset=utf-8")

    def read_text(self, key):
        blob = self.bucket.blob(self._k(key))
        try:
            return blob.download_as_bytes().decode("utf-8")
        except Exception as e:
            if _is_missing(e):
                return None
            raise

    def list(self, prefix):
        return sorted(self._strip(b.name) for b in self.client.list_blobs(self.bucket, prefix=self._k(prefix)))

    def delete(self, key):
        try:
            self.bucket.blob(self._k(key)).delete()
        except Exception as e:
            if not _is_missing(e):
                raise


def _is_missing(e: Exception) -> bool:
    """'Not found' across SDKs without importing them."""
    name = type(e).__name__
    if name in ("NoSuchKey", "ResourceNotFoundError", "NotFound", "KeyError", "FileNotFoundError"):
        return True
    code = getattr(e, "response", {}) or {}
    if isinstance(code, dict) and (code.get("Error") or {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
        return True
    return getattr(e, "status_code", None) == 404 or getattr(e, "code", None) == 404
