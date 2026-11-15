
import json
from urllib.parse import quote

import requests


class PersonalStore:
    """
    Personal Actions File Store Client.

    API:
        POST /store/{filename}?token=xxx
        GET  /store/{filename}?token=xxx
    """

    def __init__(
        self,
        base_url: str = "https://hc.lxc.nya.je:24687/",
        token: str = "",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

        self.session = requests.Session()

    def _url(self, filename: str) -> str:
        if not isinstance(filename, str) or not filename:
            raise ValueError("filename must be a non-empty string")

        if "/" in filename or "\\" in filename:
            raise ValueError("filename must not contain path separators")

        return f"{self.base_url}/store/{quote(filename, safe='')}"

    def save(self, filename: str, content: str) -> dict:
        """
        Save or overwrite a file.

        Args:
            filename: Target filename.
            content: String content to store.

        Returns:
            API response dictionary.
        """
        if not isinstance(content, str):
            raise TypeError("content must be a string")

        response = self.session.post(
            self._url(filename),
            params={"token": self.token},
            json={"content": content},
            timeout=self.timeout,
        )

        response.raise_for_status()
        return response.json()

    def load(self, filename: str) -> str:
        """
        Read a file.

        Returns:
            Raw file content as UTF-8 string.
        """
        response = self.session.get(
            self._url(filename),
            params={"token": self.token},
            timeout=self.timeout,
        )

        response.raise_for_status()
        return response.content.decode("utf-8")

    def save_json(
        self,
        filename: str,
        data,
        indent: int | None = 2,
    ) -> dict:
        """
        Serialize an object to JSON and save it.
        """
        content = json.dumps(
            data,
            ensure_ascii=False,
            indent=indent,
        )

        return self.save(filename, content)

    def load_json(self, filename: str):
        """
        Load a file and deserialize its JSON content.
        """
        return json.loads(self.load(filename))

    def exists(self, filename: str) -> bool:
        """
        Check whether a file exists.
        """
        try:
            self.load(filename)
            return True
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return False
            raise

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
