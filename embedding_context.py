"""Count input tokens with the tokenizer of a loaded LM Studio embedding model."""

import json
import sys

import lmstudio


if __name__ == "__main__":
    host, identifier = sys.argv[1:3]
    with lmstudio.Client(host) as client:
        if "--loaded-only" in sys.argv[3:]:
            # Never let the SDK auto-load a second instance during a managed request.
            model = next((handle for handle in client.embedding.list_loaded() if handle.identifier == identifier), None)
            if model is None:
                sys.exit("The managed embedding instance is no longer loaded.")
        else:
            model = client.embedding.model(identifier)
        print(json.dumps([model.count_tokens(sys.stdin.buffer.read().decode("utf-8")),
                          model.get_context_length()]))
