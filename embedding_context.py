"""Count input tokens with the tokenizer of a loaded LM Studio embedding model."""

import json
import sys

import lmstudio


if __name__ == "__main__":
    host, identifier = sys.argv[1:]
    with lmstudio.Client(host) as client:
        model = client.embedding.model(identifier)
        print(json.dumps([model.count_tokens(sys.stdin.buffer.read().decode("utf-8")),
                          model.get_context_length()]))
