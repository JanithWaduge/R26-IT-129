import hashlib
import hmac


class Pseudonymizer:
    def __init__(self, secret: str) -> None:
        if len(secret) < 32:
            raise ValueError(
                "ML pseudonym secret must contain at least 32 characters."
            )
        self._secret = secret.encode("utf-8")

    def token(self, *, namespace: str, value: str) -> str:
        message = f"{namespace}:{value}".encode("utf-8")
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()

    def learner(self, value: str) -> str:
        return self.token(namespace="learner", value=value)

    def sign(self, value: str) -> str:
        return self.token(namespace="sign", value=value)

    def row(self, value: str) -> str:
        return self.token(namespace="row", value=value)

    def event(self, value: str) -> str:
        return self.token(namespace="event", value=value)

    def session(self, value: str) -> str:
        return self.token(namespace="session", value=value)

    def question(self, value: str) -> str:
        return self.token(namespace="question", value=value)

    def split_for_learner(self, learner_group_id: str) -> str:
        split_hash = self.token(
            namespace="dataset-split", value=learner_group_id
        )
        bucket = int(split_hash[:8], 16) % 100
        if bucket < 70:
            return "train"
        if bucket < 85:
            return "validation"
        return "test"
